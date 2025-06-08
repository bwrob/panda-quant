import importlib
from product_definitions import ProductStaticBase # To check class inheritance

def import_product_class(module_name: str, class_name: str) -> type:
    """
    Dynamically imports a class from a given module.

    Args:
        module_name (str): The name of the module (e.g., 'my_project.product_defs').
        class_name (str): The name of the class to import from the module.

    Returns:
        type: The imported class object.

    Raises:
        ImportError: If the module cannot be imported.
        AttributeError: If the class cannot be found in the module.
        TypeError: If the imported object is not a class or not a subclass of ProductStaticBase.
    """
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise ImportError(f"Could not import module '{module_name}': {e}")

    # support aliases for product_type strings
    alias_map = {
        "VanillaBond":    "QuantLibBondStaticBase",
        "CallableBond":   "CallableBondStaticBase",
        "ConvertibleBond":"ConvertibleBondStaticBase",
        "EuropeanOption": "EuropeanOptionStatic",
        "MBSPool":        "MBSPoolStatic"
    }
    candidates = [class_name]
    if class_name in alias_map:
        candidates.append(alias_map[class_name])

    for cname in candidates:
        if hasattr(module, cname):
            imported_class = getattr(module, cname)
            break
    else:
        raise AttributeError(f"Class '{class_name}' (or aliases {candidates}) not found in module '{module_name}'.")

    if not isinstance(imported_class, type):
        raise TypeError(f"'{module_name}.{class_name}' is not a class.")

    # Optional: Check if it's a subclass of a specific base if needed
    if not issubclass(imported_class, ProductStaticBase):
        raise TypeError(f"Class '{class_name}' from module '{module_name}' is not a subclass of ProductStaticBase.")

    return imported_class

def import_class_from_module(module_name: str, class_name: str, expected_base_class: type = None) -> type:
    """
    Dynamically imports a class from a given module and optionally checks its inheritance.

    Args:
        module_name (str): The name of the module (e.g., 'my_project.product_defs').
        class_name (str): The name of the class to import from the module.
        expected_base_class (type, optional): If provided, checks if the imported
                                              class is a subclass of this base.

    Returns:
        type: The imported class object.

    Raises:
        ImportError: If the module cannot be imported.
        AttributeError: If the class cannot be found in the module.
        TypeError: If the imported object is not a class or (if checked) not a
                   subclass of expected_base_class.
    """
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise ImportError(f"Could not import module '{module_name}': {e}")

    try:
        imported_class = getattr(module, class_name)
    except AttributeError:
        raise AttributeError(f"Class '{class_name}' not found in module '{module_name}'.")

    if not isinstance(imported_class, type):
        raise TypeError(f"'{module_name}.{class_name}' is not a class.")

    if expected_base_class and not issubclass(imported_class, expected_base_class):
        raise TypeError(
            f"Class '{class_name}' from module '{module_name}' is not a subclass of {expected_base_class.__name__}."
        )

    return imported_class

def create_product_static_from_dict(params: dict) -> ProductStaticBase:
    """
    Factory: uses params['module_name'] and params['product_type'] (class name)
    to dynamically import and dispatch to the correct subclass.
    """
    module_name_str = params.get('module_name')
    product_class_name_str = params.get('product_type') # 'product_type' now means class name

    if not module_name_str:
        raise KeyError("Configuration must include 'module_name'.")
    if not product_class_name_str:
        raise KeyError("Configuration must include 'product_type' (string name of the class).")

    # Use the custom importer
    try:
        product_class = import_product_class(module_name_str, product_class_name_str)
    except (ImportError, AttributeError, TypeError) as e:
        # More specific error handling can be added here if needed
        raise ValueError(f"Failed to load product class '{product_class_name_str}' from module '{module_name_str}': {e}")

    # The imported class should have a 'from_dict' classmethod
    if not hasattr(product_class, 'from_dict') or not callable(getattr(product_class, 'from_dict')):
        raise NotImplementedError(f"Class '{product_class_name_str}' does not have a callable 'from_dict' classmethod.")

    return product_class.from_dict(params)