"""
NumPy Analyzer - Complete Data Analysis Toolkit
A beginner-friendly project that demonstrates NumPy operations with Object-Oriented Programming
"""

import numpy as np

class DataAnalytics:
    """
    A class for performing various data analysis operations using NumPy arrays.
    This is designed to be beginner-friendly with clear explanations.
    """
    
    def __init__(self, array=None):
        """
        Constructor to initialize the DataAnalytics object with a NumPy array.
        
        What is a constructor?
        - It's a special method that runs automatically when we create a new object
        - It sets up the initial state of the object
        - Here, we can optionally provide an array when creating the object
        """
        self._array = array  # The underscore _ means this is "private" (encapsulated)
    
    # Properties for safe access to the array (Encapsulation)
    @property
    def array(self):
        """Getter - allows safe reading of the array"""
        return self._array
    
    @array.setter
    def array(self, value):
        """Setter - allows safe modification of the array with validation"""
        if isinstance(value, np.ndarray):
            self._array = value
        else:
            raise ValueError("Input must be a NumPy array")
    
    # ==================== ARRAY CREATION METHODS ====================
    
    @classmethod
    def create_1d_array(cls, elements):
        """
        Create a 1D array from a list of elements.
        
        What is a class method?
        - It can be called without creating an object first
        - Uses @classmethod decorator
        - 'cls' refers to the class itself (like 'self' refers to the object)
        """
        print("Creating 1D Array...")
        
        # Handle both string input "1 2 3" and list input [1, 2, 3]
        if isinstance(elements, str):
            elements = list(map(float, elements.split()))
        
        # Create NumPy array
        array = np.array(elements)
        print(f"Created 1D array with shape: {array.shape}")
        return cls(array)
    
    @classmethod
    def create_2d_array(cls, rows, cols, elements):
        """
        Create a 2D array (like a table) with specified dimensions.
        """
        print("Creating 2D Array...")
        
        if isinstance(elements, str):
            elements = list(map(float, elements.split()))
        
        # Check if we have the right number of elements
        required_elements = rows * cols
        if len(elements) != required_elements:
            raise ValueError(f"Expected {required_elements} elements, but got {len(elements)}")
        
        # Reshape the 1D list into 2D
        array = np.array(elements).reshape(rows, cols)
        print(f"Created 2D array with shape: {array.shape}")
        return cls(array)
    
    @classmethod
    def create_3d_array(cls, depth, rows, cols, elements):
        """
        Create a 3D array (like a cube) with specified dimensions.
        """
        print("Creating 3D Array...")
        
        if isinstance(elements, str):
            elements = list(map(float, elements.split()))
        
        # Check if we have the right number of elements
        required_elements = depth * rows * cols
        if len(elements) != required_elements:
            raise ValueError(f"Expected {required_elements} elements, but got {len(elements)}")
        
        # Reshape into 3D
        array = np.array(elements).reshape(depth, rows, cols)
        print(f"Created 3D array with shape: {array.shape}")
        return cls(array)
    
    # ==================== ARRAY ACCESS METHODS ====================
    
    def indexing(self, row=None, col=None):
        """
        Access specific elements using indexing.
        
        What is indexing?
        - Getting a single element from an array using its position
        - Like finding a specific seat in a theater using row and column numbers
        """
        if self._array is None:
            raise ValueError("No array created yet!")
        
        print(f"Accessing element at position: row={row}, col={col}")
        
        if self._array.ndim == 1:  # 1D array
            return self._array[row] if row is not None else self._array
        
        elif self._array.ndim == 2:  # 2D array
            if row is not None and col is not None:
                return self._array[row, col]
            elif row is not None:
                return self._array[row, :]
            else:
                return self._array
        
        return self._array
    
    def slicing(self, row_start, row_end, col_start=None, col_end=None):
        """
        Extract a portion (slice) from the array.
        
        What is slicing?
        - Getting a subset of the array
        - Like cutting a piece from a cake
        - Syntax: start:end (includes start, excludes end)
        """
        if self._array is None:
            raise ValueError("No array created yet!")
        
        print(f"Slicing array: rows[{row_start}:{row_end}], columns[{col_start}:{col_end}]")
        
        if self._array.ndim == 1:
            return self._array[row_start:row_end]
        else:
            if col_start is not None and col_end is not None:
                return self._array[row_start:row_end, col_start:col_end]
            else:
                return self._array[row_start:row_end, :]
    
    # ==================== MATHEMATICAL OPERATIONS ====================
    
    def elementwise_operation(self, other_array, operation):
        """
        Perform element-wise mathematical operations.
        
        What is element-wise operation?
        - Apply operation to each corresponding element
        - Arrays must be same shape
        - Example: [1,2,3] + [4,5,6] = [5,7,9]
        """
        if self._array is None:
            raise ValueError("No array created yet!")
        
        # Convert to NumPy array if needed
        if not isinstance(other_array, np.ndarray):
            other_array = np.array(other_array)
        
        # Check if shapes match
        if self._array.shape != other_array.shape:
            raise ValueError(f"Shapes don't match! {self._array.shape} vs {other_array.shape}")
        
        # Dictionary of operations
        operations = {
            'add': np.add,
            'subtract': np.subtract,
            'multiply': np.multiply,
            'divide': np.divide
        }
        
        if operation not in operations:
            raise ValueError(f"Unknown operation: {operation}")
        
        print(f"Performing {operation} operation...")
        return operations[operation](self._array, other_array)
    
    def dot_product(self, other_array):
        """
        Calculate dot product or matrix multiplication.
        
        What is dot product?
        - For 1D: sum of products of corresponding elements
        - For 2D: proper matrix multiplication
        - Different from element-wise operations
        """
        if self._array is None:
            raise ValueError("No array created yet!")
        
        if not isinstance(other_array, np.ndarray):
            other_array = np.array(other_array)
        
        print("Calculating dot product...")
        return np.dot(self._array, other_array)
    
    # ==================== ARRAY COMBINATION AND SPLITTING ====================
    
    def concatenate_arrays(self, other_array, axis=0):
        """
        Join two arrays together.
        
        What is concatenation?
        - Joining arrays along a specified axis
        - axis=0: vertical stacking (one below another)
        - axis=1: horizontal stacking (side by side)
        """
        if self._array is None:
            raise ValueError("No array created yet!")
        
        if not isinstance(other_array, np.ndarray):
            other_array = np.array(other_array)
        
        axis_names = {0: "vertical", 1: "horizontal"}
        print(f"Concatenating arrays {axis_names.get(axis, 'along axis ' + str(axis))}...")
        
        return np.concatenate((self._array, other_array), axis=axis)
    
    def split_array(self, num_sections, axis=0):
        """
        Split array into multiple smaller arrays.
        """
        if self._array is None:
            raise ValueError("No array created yet!")
        
        print(f"Splitting array into {num_sections} sections along axis {axis}...")
        return np.array_split(self._array, num_sections, axis=axis)
    
    # ==================== SEARCH, SORT, AND FILTER ====================
    
    def search_value(self, value):
        """
        Search for specific values in the array.
        
        Returns the positions where the value is found.
        """
        if self._array is None:
            raise ValueError("No array created yet!")
        
        print(f"Searching for value: {value}")
        indices = np.where(self._array == value)
        
        if len(indices[0]) == 0:
            print(f"Value {value} not found in array")
        else:
            print(f"Value {value} found at positions: {list(zip(indices[0], indices[1])) if self._array.ndim > 1 else indices[0]}")
        
        return indices
    
    def sort_array(self, axis=-1):
        """
        Sort the array in ascending order.
        """
        if self._array is None:
            raise ValueError("No array created yet!")
        
        print(f"Sorting array along axis {axis}...")
        return np.sort(self._array, axis=axis)
    
    def filter_array(self, condition):
        """
        Filter array based on a condition.
        
        Examples of conditions:
        - > 5: keep values greater than 5
        - == 0: keep values equal to 0
        - < 10: keep values less than 10
        """
        if self._array is None:
            raise ValueError("No array created yet!")
        
        print(f"Filtering array with condition: {condition}")
        
        # Create condition based on input
        if condition.startswith('>='):
            value = float(condition[2:])
            filtered = self._array[self._array >= value]
        elif condition.startswith('<='):
            value = float(condition[2:])
            filtered = self._array[self._array <= value]
        elif condition.startswith('>'):
            value = float(condition[1:])
            filtered = self._array[self._array > value]
        elif condition.startswith('<'):
            value = float(condition[1:])
            filtered = self._array[self._array < value]
        elif condition.startswith('=='):
            value = float(condition[2:])
            filtered = self._array[self._array == value]
        elif condition.startswith('!='):
            value = float(condition[2:])
            filtered = self._array[self._array != value]
        else:
            raise ValueError("Invalid condition format. Use '>5', '<=10', '==0', etc.")
        
        print(f"Found {len(filtered)} elements matching condition")
        return filtered
    
    # ==================== AGGREGATING FUNCTIONS ====================
    
    def calculate_sum(self, axis=None):
        """Calculate sum of all elements"""
        if self._array is None:
            raise ValueError("No array created yet!")
        return np.sum(self._array, axis=axis)
    
    def calculate_mean(self, axis=None):
        """Calculate average (mean) of elements"""
        if self._array is None:
            raise ValueError("No array created yet!")
        return np.mean(self._array, axis=axis)
    
    def calculate_median(self, axis=None):
        """Calculate median (middle value)"""
        if self._array is None:
            raise ValueError("No array created yet!")
        return np.median(self._array, axis=axis)
    
    def calculate_std(self, axis=None):
        """Calculate standard deviation (spread of data)"""
        if self._array is None:
            raise ValueError("No array created yet!")
        return np.std(self._array, axis=axis)
    
    def calculate_variance(self, axis=None):
        """Calculate variance (average of squared differences from mean)"""
        if self._array is None:
            raise ValueError("No array created yet!")
        return np.var(self._array, axis=axis)
    
    def calculate_min(self, axis=None):
        """Find smallest value"""
        if self._array is None:
            raise ValueError("No array created yet!")
        return np.min(self._array, axis=axis)
    
    def calculate_max(self, axis=None):
        """Find largest value"""
        if self._array is None:
            raise ValueError("No array created yet!")
        return np.max(self._array, axis=axis)
    
    def calculate_percentile(self, percent, axis=None):
        """
        Calculate percentile.
        
        What is percentile?
        - Value below which a given percentage of observations fall
        - Example: 75th percentile means 75% of values are below this value
        """
        if self._array is None:
            raise ValueError("No array created yet!")
        return np.percentile(self._array, percent, axis=axis)
    
    # ==================== UTILITY METHODS ====================
    
    def display_array(self, title="Current Array"):
        """Display the current array in a nice format"""
        if self._array is None:
            print("No array created yet!")
            return
        
        print(f"\n{title}:")
        print(self._array)
        print(f"Shape: {self._array.shape}, Data Type: {self._array.dtype}")
    
    @staticmethod
    def validate_numeric_input(input_str):
        """
        Static method to validate numeric input.
        
        What is a static method?
        - Doesn't need access to the object or class
        - Can be called without creating an object
        - Used for utility functions
        """
        try:
            numbers = list(map(float, input_str.split()))
            return numbers
        except ValueError:
            raise ValueError("Invalid input. Please enter numbers only.")


class NumPyAnalyzer:
    """
    User Interface for the NumPy Analyzer - Beginner Friendly Version
    """
    
    def __init__(self):
        self.analyzer = None
        self.running = True
    
    def print_header(self):
        """Print a nice header"""
        print("\n" + "="*60)
        print("           🧮 NUM PY ANALYZER - BEGINNER FRIENDLY 🧮")
        print("="*60)
    
    def print_footer(self):
        """Print a nice footer"""
        print("\n" + "="*60)
        print("           💡 Tip: Arrays make data analysis easier!")
        print("="*60)
    
    def display_main_menu(self):
        """Display the main menu"""
        self.print_header()
        print("Welcome! Let's learn NumPy operations step by step!")
        print("\nWhat would you like to do?")
        print("1. 📊 Create a New Array")
        print("2. ➕➖ Mathematical Operations")
        print("3. 🔗 Combine or Split Arrays") 
        print("4. 🔍 Search, Sort, or Filter")
        print("5. 📈 Statistics & Analysis")
        print("6. ❌ Exit")
        print("\n" + "-"*40)
    
    def wait_for_enter(self):
        """Wait for user to press enter to continue"""
        input("\nPress Enter to continue...")
    
    def create_array_menu(self):
        """Menu for creating arrays"""
        print("\n" + "="*40)
        print("          📊 ARRAY CREATION")
        print("="*40)
        print("\nWhat type of array would you like to create?")
        print("1. 1D Array (like a simple list)")
        print("2. 2D Array (like a table or spreadsheet)")
        print("3. 3D Array (like a cube or multiple tables)")
        print("4. 🔙 Back to Main Menu")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            self.create_1d_array()
        elif choice == '2':
            self.create_2d_array()
        elif choice == '3':
            self.create_3d_array()
        elif choice == '4':
            return
        else:
            print("❌ Invalid choice. Please try again.")
            self.create_array_menu()
    
    def create_1d_array(self):
        """Create a 1-dimensional array"""
        print("\n--- Creating 1D Array ---")
        print("Example: Enter '1 2 3 4 5' for array [1, 2, 3, 4, 5]")
        
        try:
            elements = input("Enter numbers separated by spaces: ").strip()
            self.analyzer = DataAnalytics.create_1d_array(elements)
            self.analyzer.display_array("Your 1D Array")
            self.wait_for_enter()
        except Exception as e:
            print(f"❌ Error: {e}")
            self.wait_for_enter()
    
    def create_2d_array(self):
        """Create a 2-dimensional array"""
        print("\n--- Creating 2D Array ---")
        print("Example: 2 rows, 3 columns, elements '10 20 30 40 50 60'")
        print("Creates: [[10, 20, 30], [40, 50, 60]]")
        
        try:
            rows = int(input("Enter number of rows: "))
            cols = int(input("Enter number of columns: "))
            elements = input(f"Enter {rows * cols} numbers separated by spaces: ").strip()
            
            self.analyzer = DataAnalytics.create_2d_array(rows, cols, elements)
            self.analyzer.display_array("Your 2D Array")
            self.wait_for_enter()
        except Exception as e:
            print(f"❌ Error: {e}")
            self.wait_for_enter()
    
    def create_3d_array(self):
        """Create a 3-dimensional array"""
        print("\n--- Creating 3D Array ---")
        print("Example: 2 layers, 2 rows, 3 columns")
        print("Elements: '1 2 3 4 5 6 7 8 9 10 11 12'")
        
        try:
            depth = int(input("Enter number of layers: "))
            rows = int(input("Enter number of rows per layer: "))
            cols = int(input("Enter number of columns per layer: "))
            total = depth * rows * cols
            elements = input(f"Enter {total} numbers separated by spaces: ").strip()
            
            self.analyzer = DataAnalytics.create_3d_array(depth, rows, cols, elements)
            self.analyzer.display_array("Your 3D Array")
            self.wait_for_enter()
        except Exception as e:
            print(f"❌ Error: {e}")
            self.wait_for_enter()
    
    def math_operations_menu(self):
        """Menu for mathematical operations"""
        if not self.check_array_exists():
            return
        
        print("\n" + "="*40)
        print("          ➕ MATHEMATICAL OPERATIONS")
        print("="*40)
        print("\nChoose an operation:")
        print("1. ➕ Addition")
        print("2. ➖ Subtraction") 
        print("3. ✖️ Multiplication")
        print("4. ➗ Division")
        print("5. 🔷 Dot Product (Matrix Multiplication)")
        print("6. 🔙 Back to Main Menu")
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice in ['1', '2', '3', '4']:
            self.elementwise_operations(choice)
        elif choice == '5':
            self.dot_product_operation()
        elif choice == '6':
            return
        else:
            print("❌ Invalid choice. Please try again.")
            self.math_operations_menu()
    
    def elementwise_operations(self, operation):
        """Handle element-wise operations"""
        operations = {
            '1': ('Addition', 'add'),
            '2': ('Subtraction', 'subtract'), 
            '3': ('Multiplication', 'multiply'),
            '4': ('Division', 'divide')
        }
        
        op_name, op_code = operations[operation]
        
        print(f"\n--- {op_name} Operation ---")
        print("Element-wise means we apply the operation to each corresponding element")
        print(f"Example: [1,2,3] {op_name.lower()} [4,5,6] = [5,7,9]")
        
        try:
            # Show current array
            self.analyzer.display_array("Your Current Array")
            
            # Get second array
            elements = input(f"\nEnter numbers for second array (same size): ").strip()
            second_array = DataAnalytics.validate_numeric_input(elements)
            second_array = np.array(second_array).reshape(self.analyzer.array.shape)
            
            # Perform operation
            result = self.analyzer.elementwise_operation(second_array, op_code)
            
            # Show results
            print(f"\nSecond Array:")
            print(second_array)
            print(f"\nResult of {op_name}:")
            print(result)
            
            self.wait_for_enter()
        except Exception as e:
            print(f"❌ Error: {e}")
            self.wait_for_enter()
    
    def dot_product_operation(self):
        """Handle dot product operation"""
        print("\n--- Dot Product / Matrix Multiplication ---")
        print("Different from element-wise operations!")
        print("For matrices: number of columns in first must equal rows in second")
        
        try:
            self.analyzer.display_array("Your Current Array")
            
            if self.analyzer.array.ndim == 1:
                # 1D array - simple dot product
                elements = input("Enter numbers for second array (same size): ").strip()
                second_array = DataAnalytics.validate_numeric_input(elements)
                second_array = np.array(second_array)
                
                result = self.analyzer.dot_product(second_array)
                print(f"\nSecond Array: {second_array}")
                print(f"\nDot Product Result: {result}")
                
            else:
                # 2D array - matrix multiplication
                cols = self.analyzer.array.shape[1]
                rows2 = int(input(f"Enter rows for second matrix (must match current columns {cols}): "))
                cols2 = int(input("Enter columns for second matrix: "))
                
                elements = input(f"Enter {rows2 * cols2} numbers: ").strip()
                second_array = DataAnalytics.validate_numeric_input(elements)
                second_array = np.array(second_array).reshape(rows2, cols2)
                
                result = self.analyzer.dot_product(second_array)
                
                print(f"\nSecond Array:")
                print(second_array)
                print(f"\nMatrix Multiplication Result:")
                print(result)
            
            self.wait_for_enter()
        except Exception as e:
            print(f"❌ Error: {e}")
            self.wait_for_enter()
    
    def combine_split_menu(self):
        """Menu for combining and splitting arrays"""
        if not self.check_array_exists():
            return
        
        print("\n" + "="*40)
        print("          🔗 COMBINE & SPLIT ARRAYS")
        print("="*40)
        print("\nChoose an operation:")
        print("1. 🔗 Combine (Concatenate) Arrays")
        print("2. ✂️ Split Array into Parts") 
        print("3. 🔙 Back to Main Menu")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            self.combine_arrays()
        elif choice == '2':
            self.split_array()
        elif choice == '3':
            return
        else:
            print("❌ Invalid choice. Please try again.")
            self.combine_split_menu()
    
    def combine_arrays(self):
        """Combine two arrays"""
        print("\n--- Combining Arrays ---")
        print("We'll combine your current array with a new one")
        
        try:
            self.analyzer.display_array("Your Current Array")
            
            axis = int(input("Combine vertically (0) or horizontally (1)?: "))
            elements = input("Enter numbers for second array: ").strip()
            second_array = DataAnalytics.validate_numeric_input(elements)
            
            # Try to reshape to match dimensions
            try:
                second_array = np.array(second_array).reshape(self.analyzer.array.shape)
            except:
                # If reshaping fails, use as is
                second_array = np.array(second_array)
            
            result = self.analyzer.concatenate_arrays(second_array, axis=axis)
            
            print(f"\nSecond Array:")
            print(second_array)
            print(f"\nCombined Result:")
            print(result)
            
            self.wait_for_enter()
        except Exception as e:
            print(f"❌ Error: {e}")
            self.wait_for_enter()
    
    def split_array(self):
        """Split array into parts"""
        print("\n--- Splitting Array ---")
        
        try:
            self.analyzer.display_array("Your Current Array")
            
            num_parts = int(input("How many parts to split into?: "))
            result = self.analyzer.split_array(num_parts)
            
            print(f"\nSplit into {num_parts} parts:")
            for i, part in enumerate(result):
                print(f"Part {i+1}:")
                print(part)
                print()
            
            self.wait_for_enter()
        except Exception as e:
            print(f"❌ Error: {e}")
            self.wait_for_enter()
    
    def search_sort_filter_menu(self):
        """Menu for search, sort and filter operations"""
        if not self.check_array_exists():
            return
        
        print("\n" + "="*40)
        print("          🔍 SEARCH, SORT & FILTER")
        print("="*40)
        print("\nChoose an operation:")
        print("1. 🔎 Search for Values")
        print("2. 📊 Sort Array") 
        print("3. 🎯 Filter Array")
        print("4. 🔙 Back to Main Menu")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            self.search_values()
        elif choice == '2':
            self.sort_array()
        elif choice == '3':
            self.filter_array()
        elif choice == '4':
            return
        else:
            print("❌ Invalid choice. Please try again.")
            self.search_sort_filter_menu()
    
    def search_values(self):
        """Search for specific values"""
        print("\n--- Searching Array ---")
        
        try:
            self.analyzer.display_array("Your Current Array")
            
            value = float(input("Enter value to search for: "))
            self.analyzer.search_value(value)
            
            self.wait_for_enter()
        except Exception as e:
            print(f"❌ Error: {e}")
            self.wait_for_enter()
    
    def sort_array(self):
        """Sort the array"""
        print("\n--- Sorting Array ---")
        print("Sorting rearranges elements in ascending order")
        
        try:
            self.analyzer.display_array("Original Array")
            
            result = self.analyzer.sort_array()
            
            print(f"\nSorted Array:")
            print(result)
            
            self.wait_for_enter()
        except Exception as e:
            print(f"❌ Error: {e}")
            self.wait_for_enter()
    
    def filter_array(self):
        """Filter array based on condition"""
        print("\n--- Filtering Array ---")
        print("Examples of conditions:")
        print("'>5' - keep values greater than 5")
        print("'<=10' - keep values less than or equal to 10") 
        print("'==0' - keep values equal to 0")
        
        try:
            self.analyzer.display_array("Your Current Array")
            
            condition = input("Enter condition (e.g., '>5', '<=10', '==0'): ").strip()
            result = self.analyzer.filter_array(condition)
            
            print(f"\nFiltered Array ({condition}):")
            print(result)
            
            self.wait_for_enter()
        except Exception as e:
            print(f"❌ Error: {e}")
            self.wait_for_enter()
    
    def statistics_menu(self):
        """Menu for statistical operations"""
        if not self.check_array_exists():
            return
        
        print("\n" + "="*40)
        print("          📈 STATISTICS & ANALYSIS")
        print("="*40)
        print("\nChoose a statistical operation:")
        print("1. ∑ Sum")
        print("2. μ Mean (Average)")
        print("3. 📊 Median (Middle Value)")
        print("4. σ Standard Deviation")
        print("5. 📊 Variance")
        print("6. 📉 Minimum Value")
        print("7. 📈 Maximum Value")
        print("8. 📊 Percentile")
        print("9. 🔙 Back to Main Menu")
        
        choice = input("\nEnter your choice (1-9): ").strip()
        
        operations = {
            '1': ('Sum', self.analyzer.calculate_sum),
            '2': ('Mean', self.analyzer.calculate_mean),
            '3': ('Median', self.analyzer.calculate_median),
            '4': ('Standard Deviation', self.analyzer.calculate_std),
            '5': ('Variance', self.analyzer.calculate_variance),
            '6': ('Minimum', self.analyzer.calculate_min),
            '7': ('Maximum', self.analyzer.calculate_max),
            '8': self.calculate_percentile
        }
        
        if choice in operations:
            try:
                self.analyzer.display_array("Your Array")
                
                if choice == '8':
                    operations[choice]()
                else:
                    op_name, op_func = operations[choice]
                    result = op_func()
                    print(f"\n{op_name}: {result}")
                
                self.wait_for_enter()
            except Exception as e:
                print(f"❌ Error: {e}")
                self.wait_for_enter()
        elif choice == '9':
            return
        else:
            print("❌ Invalid choice. Please try again.")
            self.statistics_menu()
    
    def calculate_percentile(self):
        """Calculate percentile"""
        print("\n--- Calculating Percentile ---")
        print("Example: 75th percentile means 75% of values are below this number")
        
        try:
            percent = float(input("Enter percentile (0-100): "))
            result = self.analyzer.calculate_percentile(percent)
            
            print(f"\n{percent}th Percentile: {result}")
            
            self.wait_for_enter()
        except Exception as e:
            print(f"❌ Error: {e}")
            self.wait_for_enter()
    
    def check_array_exists(self):
        """Check if array exists, show message if not"""
        if self.analyzer is None or self.analyzer.array is None:
            print("❌ Please create an array first (Option 1 in main menu)")
            self.wait_for_enter()
            return False
        return True
    
    def run(self):
        """Main program loop"""
        print("🚀 Starting NumPy Analyzer...")
        print("📚 This program will help you learn NumPy operations!")
        
        while self.running:
            try:
                self.display_main_menu()
                choice = input("\nEnter your choice (1-6): ").strip()
                
                if choice == '1':
                    self.create_array_menu()
                elif choice == '2':
                    self.math_operations_menu()
                elif choice == '3':
                    self.combine_split_menu()
                elif choice == '4':
                    self.search_sort_filter_menu()
                elif choice == '5':
                    self.statistics_menu()
                elif choice == '6':
                    print("\n🎉 Thank you for using NumPy Analyzer!")
                    print("🌟 Keep practicing and happy coding!")
                    break
                else:
                    print("❌ Invalid choice. Please enter a number between 1-6.")
                    self.wait_for_enter()
                    
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye! Thanks for using NumPy Analyzer!")
                break
            except Exception as e:
                print(f"❌ An error occurred: {e}")
                self.wait_for_enter()


# ==================== MAIN PROGRAM ====================

if __name__ == "__main__":
    # Check if NumPy is installed
    try:
        import numpy as np
        print("✅ NumPy is installed and ready!")
    except ImportError:
        print("❌ NumPy is not installed. Please install it using:")
        print("   pip install numpy")
        exit(1)
    
    # Create and run the application
    app = NumPyAnalyzer()
    app.run()