"""
ROCmForge Studio — AST-Based CUDA Primitive Classifier

Deterministic, tree-sitter AST classifier that detects:
  • GEMM (matmul patterns, cuBLAS calls)
  • Reduction (sum, max, atomicAdd, warp-level reductions)
  • Elementwise (pointwise / element-wise ops)
  • Fused Matmul (GEMM + Activation)
"""

import re
from typing import Any, Dict, List, Optional
import tree_sitter
import tree_sitter_cpp

# Initialize parser
parser = tree_sitter.Parser(tree_sitter.Language(tree_sitter_cpp.language()))

# Fallback definitions
_DTYPE_MAP = {
    "half": "half", "__half": "half", "float16": "half", "fp16": "half",
    "float": "float", "fp32": "float",
    "double": "double", "fp64": "double",
    "int": "int", "int32_t": "int"
}

def _traverse_ast(node, target_types=None, target_text=None, results=None):
    """Recursively search AST for specific node types or source text matches."""
    if results is None:
        results = []
    
    match = True
    if target_types and node.type not in target_types:
        match = False
    
    if target_text and match:
        # Extract source bytes if needed for text matching
        text = node.text.decode('utf8')
        if not re.search(target_text, text, re.IGNORECASE):
            match = False
            
    if match and (target_types or target_text):
         results.append(node)

    for child in node.children:
        _traverse_ast(child, target_types, target_text, results)
        
    return results

def _detect_primitive_and_pattern_ast(code: str, root_node) -> tuple:
    primitive = "elementwise"
    pattern = "vectorized"
    
    # Check for library calls first
    calls = _traverse_ast(root_node, target_types=["call_expression"])
    for call in calls:
        call_text = call.text.decode('utf8').lower()
        if "cublas" in call_text or "rocblas" in call_text or "gemm" in call_text:
            return "gemm", "tiled_shared"
        elif "conv" in call_text or "cudnn" in call_text:
            return "conv", "direct_conv"
            
    # Check memory access and syncthreads for patterns
    shared_decls = _traverse_ast(root_node, target_text=r'\b(__shared__|shared)\b')
    syncthreads = _traverse_ast(root_node, target_text=r'__syncthreads')
    
    # Check for loops (often indicates GEMM or Reduction)
    for_loops = _traverse_ast(root_node, target_types=["for_statement"])
    
    # Check for atomics/shuffles (reduction)
    shuffles = _traverse_ast(root_node, target_text=r'__shfl|warp_reduce|reduce')
    atomics = _traverse_ast(root_node, target_text=r'atomicAdd|atomicMax')
    
    # Check for specific Math / Activations
    activations = _traverse_ast(root_node, target_text=r'\brelu\b|\bsigmoid\b|>\s*0\s*\?')
    exponents = _traverse_ast(root_node, target_text=r'\bexpf\b|\bexp\b')
    sqrts = _traverse_ast(root_node, target_text=r'\brsqrtf\b|\bsqrt\b|\brsqrt\b')
    rngs = _traverse_ast(root_node, target_text=r'\bseed\b|\brand\b')
    
    # Classify based on AST features
    # 1. Attention (typically has exp/softmax + matrix traits)
    if exponents and sum(1 for p in _traverse_ast(root_node, target_types=["parameter_declaration"]) if b'Q' in p.text or b'K' in p.text or b'V' in p.text) >= 2:
        return "attention", "flash_attention"
    
    # 2. Softmax (typically has exp, reduce max, reduce sum)
    elif exponents and shuffles and len(for_loops) >= 2:
        return "softmax", "fused_softmax_reduce"
        
    # 3. LayerNorm (typically has rsqrt, mean/var reductions)
    elif sqrts and len(for_loops) >= 2 and shuffles:
        return "layernorm", "fused_layernorm"
        
    # 4. Conv (nested loops over kernel size)
    elif len(for_loops) >= 4: # batch, channels, kernel_h, kernel_w
        return "conv", "direct_conv"
        
    # 5. Dropout (RNG based element-wise mutator)
    elif rngs and _traverse_ast(root_node, target_text=r'\bdrop\b'):
        return "dropout", "fused_dropout"
        
    # 6. Fallback Standard Primitives
    elif shuffles or atomics:
        primitive = "reduction"
        pattern = "wavefront_reduce" if shuffles else "atomic_reduce"
    elif len(for_loops) >= 1 and (shared_decls or syncthreads or len(for_loops) >= 2):
        if activations:
            primitive = "fused_matmul"
            pattern = "fused_relu"
        else:
            primitive = "gemm"
            pattern = "tiled_shared" if shared_decls else "basic_gemm"
    else:
        # Elementwise
        primitive = "elementwise"
        vector_types = _traverse_ast(root_node, target_text=r'float4|float2|double2|int4')
        pattern = "vectorized" if vector_types else "scalar"
        
    return primitive, pattern

def _extract_dims_ast(code: str, root_node) -> Dict[str, int]:
    """Extract dimensions from basic assignments or params in AST."""
    dims = {}
    
    # Check for parameter declarations that look like dimensions (M, N, K, size)
    params = _traverse_ast(root_node, target_types=["parameter_declaration"])
    
    # Also fallback to regex for simple assignments since AST for unresolved C++ might be messy
    dim_pattern = re.compile(r'(?:M|N|K|rows|cols|width|height|dim|size)\s*=\s*(\d+)', re.IGNORECASE)
    for m in dim_pattern.finditer(code):
        name = m.group(0).split("=")[0].strip().upper()
        dims[name] = int(m.group(1))
        
    return dims

def _detect_dtype_ast(root_node) -> str:
    """Find primary floating/integer type in AST."""
    types = _traverse_ast(root_node, target_types=["primitive_type", "type_identifier"])
    
    for t in types:
        text = t.text.decode('utf8').lower()
        if text in _DTYPE_MAP:
            return _DTYPE_MAP[text]
            
    return "float"

def classify(code: str) -> Dict[str, Any]:
    """
    Classify *code* as gemm / reduction / elementwise / fused_matmul.

    Returns
    -------
    dict
        primitive : str   — "gemm", "reduction", "elementwise", or "fused_matmul"
        dtype     : str   — detected data type
        shape     : str   — shape format (e.g. "M=1024, N=1024, K=1024")
        pattern   : str   — detected semantic pattern
        meta      : dict  — additional metadata (dims)
    """
    # Parse code into AST
    tree = parser.parse(bytes(code, 'utf8'))
    root = tree.root_node
    
    primitive, pattern = _detect_primitive_and_pattern_ast(code, root)
    dtype = _detect_dtype_ast(root)
    dims = _extract_dims_ast(code, root)

    # Fallback dims when none detected
    if not dims:
        if primitive in ("gemm", "fused_matmul"):
            dims = {"M": 1024, "N": 1024, "K": 1024}
        elif primitive == "reduction":
            dims = {"N": 1024}
        else:
            dims = {"N": 1024}

    shape = ", ".join(f"{k}={v}" for k, v in dims.items())

    return {
        "primitive": primitive,
        "dtype": dtype,
        "shape": shape,
        "pattern": pattern,
        "meta": {
            "dims": dims,
            "pattern": pattern
        },
    }


