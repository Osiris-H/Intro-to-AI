############################################################
## CSC 384, Intro to AI, University of Toronto.
## Assignment 3 Starter Code
## v1.1
## Changes:
##   v1.1: updated the comments in kropki_model. 
##         the second return value should be a 2d list of variables.
############################################################

from board import *
from cspbase import *


def kropki_model(board):
    """
    Create a CSP for a Kropki Sudoku Puzzle given a board of dimension.

    If a variable has an initial value, its domain should only contain the initial value.
    Otherwise, the variable's domain should contain all possible values (1 to dimension).

    We will encode all the constraints as binary constraints.
    Each constraint is represented by a list of tuples, representing the values that
    satisfy this constraint. (This is the table representation taught in lecture.)

    Remember that a Kropki sudoku has the following constraints.
    - Row constraint: every two cells in a row must have different values.
    - Column constraint: every two cells in a column must have different values.
    - Cage constraint: every two cells in a 2x3 cage (for 6x6 puzzle) 
            or 3x3 cage (for 9x9 puzzle) must have different values.
    - Black dot constraints: one value is twice the other value.
    - White dot constraints: the two values are consecutive (differ by 1).

    Make sure that you return a 2D list of variables separately. 
    Once the CSP is solved, we will use this list of variables to populate the solved board.
    Take a look at csprun.py for the expected format of this 2D list.

    :returns: A CSP object and a list of variables.
    :rtype: CSP, List[List[Variable]]

    """

    dim = board.dimension
    variables = create_variables(dim)
    for var in variables:
        var.add_domain_values(create_variables(dim))
    csp = CSP("CSP", variables)

    constraints = []
    sat_tuples = satisfying_tuples_difference_constraints(dim)
    sat_tuples_w = satisfying_tuples_white_dots(dim)
    sat_tuples_b = satisfying_tuples_black_dots(dim)
    constraints.extend(create_row_and_col_constraints(dim, sat_tuples, variables))
    constraints.extend(create_cage_constraints(dim, sat_tuples, variables))
    constraints.extend(create_dot_constraints(dim, board.dots, sat_tuples_w, sat_tuples_b, variables))
    for const in constraints:
        csp.add_constraint(const)

    return csp, [variables[i*dim: (i+1)*dim] for i in range(dim)]
    

def create_initial_domain(dim):
    """
    Return a list of values for the initial domain of any unassigned variable.
    [1, 2, ..., dimension]

    :param dim: board dimension
    :type dim: int

    :returns: A list of values for the initial domain of any unassigned variable.
    :rtype: List[int]
    """

    return list(range(1, dim + 1))


def create_variables(dim):
    """
    Return a list of variables for the board.

    We recommend that your name each variable Var(row, col).

    :param dim: Size of the board
    :type dim: int

    :returns: A list of variables, one for each cell on the board
    :rtype: List[Variables]
    """

    return [f"Var({row}, {col})" for row in range(1, dim+1) for col in range(1, dim+1)]

    
def satisfying_tuples_difference_constraints(dim):
    """
    Return a list of satifying tuples for binary difference constraints.

    :param dim: Size of the board
    :type dim: int

    :returns: A list of satifying tuples
    :rtype: List[(int,int)]
    """

    return [(i, j) for i in range(1, dim + 1) for j in range(1, dim + 1) if i != j]
  
  
def satisfying_tuples_white_dots(dim):
    """
    Return a list of satifying tuples for white dot constraints.

    :param dim: Size of the board
    :type dim: int

    :returns: A list of satifying tuples
    :rtype: List[(int,int)]
    """

    return [(i, j) for i in range(1, dim+1) for j in range(1, dim+1) if abs(i - j) == 1]

  
def satisfying_tuples_black_dots(dim):
    """
    Return a list of satifying tuples for black dot constraints.

    :param dim: Size of the board
    :type dim: int

    :returns: A list of satifying tuples
    :rtype: List[(int,int)]
    """

    return [(i, j) for i in range(1, dim+1) for j in range(1, dim+1) if i == 2*j or j == 2*i]


def create_row_and_col_constraints(dim, sat_tuples, variables):
    """
    Create and return a list of binary all-different row/column constraints.

    :param dim: Size of the board
    :type dim: int

    :param sat_tuples: A list of domain value pairs (value1, value2) such that 
        the two values in each tuple are different.
    :type sat_tuples: List[(int, int)]

    :param variables: A list of all the variables in the CSP
    :type variables: List[Variable]
        
    :returns: A list of binary all-different constraints
    :rtype: List[Constraint]
    """

    constraints = []
    # Constraints for rows
    for i in range(dim):
        for j in range(dim):
            for k in range(j + 1, dim):
                var1 = variables[i * dim + j]
                var2 = variables[i * dim + k]
                name = f"Row{i}_{var1.name}_{var2.name}"
                constraint = Constraint(name, [var1, var2])
                constraint.add_satisfying_tuples(sat_tuples)
                constraints.append(constraint)

    # Constraints for columns
    for j in range(dim):
        for i in range(dim):
            for k in range(i + 1, dim):
                var1 = variables[i * dim + j]
                var2 = variables[k * dim + j]
                name = f"Col{j}_{var1.name}_{var2.name}"
                constraint = Constraint(name, [var1, var2])
                constraint.add_satisfying_tuples(sat_tuples)
                constraints.append(constraint)

    return constraints
    
    
def create_cage_constraints(dim, sat_tuples, variables):
    """
    Create and return a list of binary all-different constraints for all cages.

    :param dim: Size of the board
    :type dim: int

    :param sat_tuples: A list of domain value pairs (value1, value2) such that 
        the two values in each tuple are different.
    :type sat_tuples: List[(int, int)]

    :param variables: A list of all the variables in the CSP
    :type variables: List[Variable]
        
    :returns: A list of binary all-different constraints
    :rtype: List[Constraint]
    """

    def pairwise_cage_constraints(indices, sat_tuples, variables):
        constraints = []
        for index1 in range(len(indices)):
            for index2 in range(index1 + 1, len(indices)):
                var1 = variables[indices[index1]]
                var2 = variables[indices[index2]]
                const_name = f"Cage_{var1.name}_{var2.name}"
                constraint = Constraint(const_name, [var1, var2])
                constraint.add_satisfying_tuples(sat_tuples)
                constraints.append(constraint)
        return constraints

    constraints = []
    subgrid_rows, subgrid_cols = (3, 2) if dim == 6 else (3, 3)

    for cage_row in range(0, dim, subgrid_rows):
        for cage_col in range(0, dim, subgrid_cols):
            indices = [(i * dim + j) for i in range(cage_row, cage_row + subgrid_rows)
                       for j in range(cage_col, cage_col + subgrid_cols)]
            constraints.extend(pairwise_cage_constraints(indices, sat_tuples, variables))

    return constraints


def create_dot_constraints(dim, dots, white_tuples, black_tuples, variables):
    """
    Create and return a list of binary constraints, one for each dot.

    :param dim: Size of the board
    :type dim: int
    
    :param dots: A list of dots, each dot is a Dot object.
    :type dots: List[Dot]

    :param white_tuples: A list of domain value pairs (value1, value2) such that 
        the two values in each tuple satisfy the white dot constraint.
    :type white_tuples: List[(int, int)]
    
    :param black_tuples: A list of domain value pairs (value1, value2) such that 
        the two values in each tuple satisfy the black dot constraint.
    :type black_tuples: List[(int, int)]

    :param variables: A list of all the variables in the CSP
    :type variables: List[Variable]
        
    :returns: A list of binary dot constraints
    :rtype: List[Constraint]
    """

    constraints = []
    for dot in dots:
        # find the two variables affected by the dot
        var1_name = f"Var({dot.cell_row}, {dot.cell_col})"
        if dot.location:
            var2_name = f"Var({dot.cell_row}, {dot.cell_col+1})"
        else:
            var2_name = f"Var({dot.cell_row+1}, {dot.cell_col})"
        var1 = None
        var2 = None
        for variable in variables:
            if variable.name == var1_name:
                var1 = variable
            elif variable.name == var2_name:
                var2 = variable
            if var1 is not None and var2 is not None:
                break
        assert var1 is not None and var2 is not None, "Variables not Found"

        if dot.color == CHAR_WHITE:
            const_name = f"Dot_W({dot.cell_row}, {dot.cell_col})"
            constraint = Constraint(const_name, [var1, var2])
            constraint.add_satisfying_tuples(white_tuples)
        else:
            const_name = f"Dot_B({dot.cell_row}, {dot.cell_col})"
            constraint = Constraint(const_name, [var1, var2])
            constraint.add_satisfying_tuples(black_tuples)
        constraints.append(constraint)

    return constraints

