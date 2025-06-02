import json
from datetime import date
import QuantLib as ql
import numpy as np

# It's good practice to import ProductStaticBase if it's explicitly checked,
# but since we are using hasattr, it's not strictly necessary for this function.
# from product_definitions import ProductStaticBase # Example if needed

def custom_json_serializer(obj):
    """
    Custom serializer for json.dumps that handles ProductStaticBase instances,
    QuantLib objects, and NumPy arrays.
    """
    # Check if the object has a to_dict method (covers ProductStaticBase and derivatives)
    if hasattr(obj, 'to_dict') and callable(obj.to_dict):
        return obj.to_dict()
    # Convert datetime.date objects to ISO format string
    elif isinstance(obj, date):
        return obj.isoformat()
    # Handle QuantLib specific types
    elif isinstance(obj, ql.Date):
         return date(obj.year(), obj.month(), obj.day()).isoformat()
    elif isinstance(obj, ql.Calendar):
        return obj.name() # Represent calendar by its name string
    elif isinstance(obj, ql.DayCounter):
        return obj.name() # Represent day counter by its name string
    elif isinstance(obj, ql.Exercise):
        # This depends on the Exercise type. For EuropeanExercise, you might serialize the expiry date.
        if isinstance(obj, ql.EuropeanExercise):
            return {'type': 'EuropeanExercise', 'expiry_date': date(obj.date().year(), obj.date().month(), obj.date().day()).isoformat()}
        # Add handling for other exercise types if necessary
        else:
            # Fallback or raise error for unhandled QuantLib Exercise types
            return {'type': obj.__class__.__name__, 'details': 'Unhandled QL Exercise type'}
    elif isinstance(obj, ql.QuoteHandle):
        if obj.empty():
            return None # Or some other representation for an empty handle
        return obj.value() # Serialize the quote's value
    elif isinstance(obj, ql.YieldTermStructureHandle):
        if obj.empty():
            return None
        # Serializing a full term structure is complex.
        # For now, return a placeholder or key info if available.
        # Example: return {'type': 'YieldTermStructure', 'reference_date': obj.referenceDate().ISO()}
        # This might need more sophisticated handling depending on use case.
        return {'type': 'YieldTermStructureHandle', 'details': 'Complex object, not fully serialized'}

    # Handle NumPy arrays
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    # Handle basic NumPy number types
    elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                        np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)

    # If other non-serializable objects (like other QuantLib objects) are *not*
    # handled by the to_dict methods or above specific handlers, they will still cause errors.
    # Ensure your to_dict methods convert all custom/QL objects to serializable types.
    else:
        # Let the default encoder attempt to handle other types, or raise TypeError
        try:
            return json.JSONEncoder.default(None, obj)
        except TypeError:
            raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable without custom handling.")
