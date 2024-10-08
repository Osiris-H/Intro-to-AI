############################################################
## CSC 384, Intro to AI, University of Toronto.
## Assignment 3 Starter Code
## v1.0
##
############################################################


def prop_FC(csp, last_assigned_var=None):
    """
    This is a propagator to perform forward checking. 

    First, collect all the relevant constraints.
    If the last assigned variable is None, then no variable has been assigned 
    and we are performing propagation before search starts.
    In this case, we will check all the constraints.
    Otherwise, we will only check constraints involving the last assigned variable.

    Among all the relevant constraints, focus on the constraints with one unassigned variable. 
    Consider every value in the unassigned variable's domain, if the value violates 
    any constraint, prune the value. 

    :param csp: The CSP problem
    :type csp: CSP
        
    :param last_assigned_var: The last variable assigned before propagation.
        None if no variable has been assigned yet (that is, we are performing 
        propagation before search starts).
    :type last_assigned_var: Variable

    :returns: The boolean indicates whether forward checking is successful.
        The boolean is False if at least one domain becomes empty after forward checking.
        The boolean is True otherwise.
        Also returns a list of variable and value pairs pruned. 
    :rtype: boolean, List[(Variable, Value)]
    """

    def check_arc_constraint(var, var_list, value, sup_tuples):
        for sup_tuple in sup_tuples:
            if all(var_list[idx].in_cur_domain(val) for idx, val in enumerate(sup_tuple)):
                return None

        var.prune_value(value)
        return var, value

    pruned_list = []

    if last_assigned_var is None:
        constraints = [const for const in csp.cons if const.get_num_unassigned_vars() == 1]
    else:
        constraints = [const for const in csp.get_cons_with_var(last_assigned_var)
                       if const.get_num_unassigned_vars() == 1]

    for const in constraints:
        var = const.get_unassigned_vars()[0]
        for value in var.cur_domain():
            if (var, value) in const.sup_tuples:
                pruned_tuple = check_arc_constraint(var, const.scope, value, const.sup_tuples[(var, value)])
            else:
                var.prune_value(value)
                pruned_tuple = (var, value)

            if pruned_tuple is not None:
                pruned_list.append(pruned_tuple)
                if var.cur_domain_size() == 0:
                    return False, pruned_list

    return True, pruned_list


def prop_AC3(csp, last_assigned_var=None):
    """
    This is a propagator to perform the AC-3 algorithm.

    Keep track of all the constraints in a queue (list). 
    If the last_assigned_var is not None, then we only need to 
    consider constraints that involve the last assigned variable.

    For each constraint, consider every variable in the constraint and 
    every value in the variable's domain.
    For each variable and value pair, prune it if it is not part of 
    a satisfying assignment for the constraint. 
    Finally, if we have pruned any value for a variable,
    add other constraints involving the variable back into the queue.

    :param csp: The CSP problem
    :type csp: CSP
        
    :param last_assigned_var: The last variable assigned before propagation.
        None if no variable has been assigned yet (that is, we are performing 
        propagation before search starts).
    :type last_assigned_var: Variable

    :returns: a boolean indicating if the current assignment satisifes 
        all the constraints and a list of variable and value pairs pruned. 
    :rtype: boolean, List[(Variable, Value)]
    """

    def revise(var, var_list, value, sup_tuples):
        for sup_tuple in sup_tuples:
            if all(var_list[idx].in_cur_domain(val) for idx, val in enumerate(sup_tuple)):
                return None

        var.prune_value(value)
        return var, value

    pruned_list = []

    if last_assigned_var is not None:
        constraints = csp.get_cons_with_var(last_assigned_var)
    else:
        constraints = csp.cons

    while constraints:
        const = constraints.pop(0)
        for var in const.scope:
            for value in var.cur_domain():
                if (var, value) in const.sup_tuples:
                    pruned_tuple = revise(var, const.scope, value, const.sup_tuples[(var, value)])
                else:
                    var.prune_value(value)
                    pruned_tuple = (var, value)

                if pruned_tuple is not None:
                    pruned_list.append(pruned_tuple)
                    if var.cur_domain_size() == 0:
                        return False, pruned_list
                    for other_const in csp.get_cons_with_var(var):
                        if other_const != const and other_const not in constraints:
                            constraints.append(other_const)

    return True, pruned_list


def ord_mrv(csp):
    """
    Implement the Minimum Remaining Values (MRV) heuristic.
    Choose the next variable to assign based on MRV.

    If there is a tie, we will choose the first variable. 

    :param csp: A CSP problem
    :type csp: CSP

    :returns: the next variable to assign based on MRV

    """

    variables = csp.vars
    target = None
    min_size = float('inf')
    for var in variables:
        if not var.is_assigned():
            size = var.cur_domain_size()
            if size < min_size:
                target = var
                min_size = size

    return target



###############################################################################
# Do not modify the prop_BT function below
###############################################################################


def prop_BT(csp, last_assigned_var=None):
    """
    This is a basic propagator for plain backtracking search.

    Check if the current assignment satisfies all the constraints.
    Note that we only need to check all the fully instantiated constraints 
    that contain the last assigned variable.
    
    :param csp: The CSP problem
    :type csp: CSP

    :param last_assigned_var: The last variable assigned before propagation.
        None if no variable has been assigned yet (that is, we are performing 
        propagation before search starts).
    :type last_assigned_var: Variable

    :returns: a boolean indicating if the current assignment satisifes all the constraints 
        and a list of variable and value pairs pruned. 
    :rtype: boolean, List[(Variable, Value)]

    """
    
    # If we haven't assigned any variable yet, return true.
    if not last_assigned_var:
        return True, []
        
    # Check all the constraints that contain the last assigned variable.
    for c in csp.get_cons_with_var(last_assigned_var):

        # All the variables in the constraint have been assigned.
        if c.get_num_unassigned_vars() == 0:

            # get the variables
            vars = c.get_scope() 

            # get the list of values
            vals = []
            for var in vars: #
                vals.append(var.get_assigned_value())

            # check if the constraint is satisfied
            if not c.check(vals): 
                return False, []

    return True, []
