############################################################
## CSC 384, Intro to AI, University of Toronto.
## Assignment 4 Starter Code
## v1.1
## - removed the example in ve since it is misleading.
############################################################

from bnetbase import Variable, Factor, BN
import csv


def generate_combs(scope):
    if not scope:
        return [[]]

    sub_combs = generate_combs(scope[1:])

    combinations = []
    for val in scope[0].domain():
        for sub_comb in sub_combs:
            combined_comb = [val] + sub_comb
            combinations.append(combined_comb)

    return combinations


def normalize(factor):
    '''
    Normalize the factor such that its values sum to 1.
    Do not modify the input factor.

    :param factor: a Factor object. 
    :return: a new Factor object resulting from normalizing factor.
    '''

    norm_factor = Factor(factor.name, factor.get_scope())
    total_sum = sum(factor.values)
    assert total_sum != 0, "Sum of probabilities is zero."
    norm_values = [value / total_sum for value in factor.values]
    norm_factor.values = norm_values
    return norm_factor


def restrict(factor, variable, value):
    '''
    Restrict a factor by assigning value to variable.
    Do not modify the input factor.

    :param factor: a Factor object.
    :param variable: the variable to restrict.
    :param value: the value to restrict the variable to
    :return: a new Factor object resulting from restricting variable to value.
             This new factor no longer has variable in it.
    ''' 

    new_scope = []
    idx = None
    for var in factor.get_scope():
        if var.name == variable.name:
            idx = factor.scope.index(var)
        else:
            new_scope.append(var)
    assert idx is not None, f"{variable.name} not in scope."
    new_factor = Factor(f"Res_{factor.name}_{variable.name}", new_scope)
    combinations = generate_combs(factor.get_scope())

    new_values = []
    for comb in combinations:
        if comb[idx] == value:
            new_values.append(factor.get_value(comb))
    new_factor.values = new_values

    return new_factor


def sum_out(factor, variable):
    '''
    Sum out a variable variable from factor factor.
    Do not modify the input factor.

    :param factor: a Factor object.
    :param variable: the variable to sum out.
    :return: a new Factor object resulting from summing out variable from the factor.
             This new factor no longer has variable in it.
    '''       

    new_scope = []
    idx = None
    for var in factor.get_scope():
        if var.name == variable.name:
            idx = factor.scope.index(var)
        else:
            new_scope.append(var)
    assert idx is not None, f"{variable.name} not in scope."

    new_factor = Factor(f"Sum_{factor.name}_{variable.name}", new_scope)
    combinations = generate_combs(new_scope)

    new_values = []
    for comb in combinations:
        prob_sum = 0
        for val in variable.domain():
            full_comb = list(comb)
            full_comb.insert(idx, val)
            prob_sum += factor.get_value(full_comb)
        new_values.append(prob_sum)
    new_factor.values = new_values

    return new_factor


def multiply(factor_list):
    '''
    Multiply a list of factors together.
    Do not modify any of the input factors. 

    :param factor_list: a list of Factor objects.
    :return: a new Factor object resulting from multiplying all the factors in factor_list.
    '''

    new_scope = []
    var_set = set()
    for factor in factor_list:
        for var in factor.get_scope():
            if var not in var_set:
                var_set.add(var)
                new_scope.append(var)
    new_name = f"Mul_{'_'.join([factor.name for factor in factor_list])}"
    new_factor = Factor(new_name, new_scope)
    combinations = generate_combs(new_scope)
    new_values = []
    for comb in combinations:
        prob = 1.0
        for factor in factor_list:
            relevant_comb = [comb[new_scope.index(var)] for var in factor.get_scope()]
            prob *= factor.get_value(relevant_comb)
        new_values.append(prob)
    new_factor.values = new_values
    return new_factor


def min_fill_ordering(factor_list, variable_query):
    '''
    This function implements The Min Fill Heuristic. We will use this heuristic to determine the order 
    to eliminate the hidden variables. The Min Fill Heuristic says to eliminate next the variable that 
    creates the factor of the smallest size. If there is a tie, choose the variable that comes first 
    in the provided order of factors in factor_list.

    Here is an example.
    Consider our complete Holmes network. Suppose that we are given a list of factors for the variables 
    in this order: P(E), P(B), P(A|B, E), P(G|A), and P(W|A). Assume that our query variable is Earthquake. 
    Among the other variables, which one should we eliminate first based on the Min Fill Heuristic? 

    - Eliminating B creates a factor of 2 variables (A and E).
    - Eliminating A creates a factor of 4 variables (E, B, G and W).
    - Eliminating G creates a factor of 1 variable (A).
    - Eliminating W creates a factor of 1 variable (A).

    In this case, G and W tie for the best variable to be eliminated first since eliminating each variable 
    creates a factor of 1 variable only. Based on our tie-breaking rule, we should choose G since it comes 
    before W in the list of factors provided.

    This function returns a list of the variables based on the min fill heuristic.
    Each variable in the returned list should come from the scopes of the factors in factor_list.
    The returned list of variables should not contain the variable_query.

    The returned list is determined iteratively.
    First, determine the size of the resulting factor when eliminating each variable from the factor_list.
    The size of the resulting factor is the number of variables in the factor.
    Then the first variable in the returned list should be the variable that results in the factor 
    of the smallest size. If there is a tie, choose the variable whose name comes first in alphabetical order.
    For example, for the complete Holmes network, we would choose B to be the first variable in the returned list.

    Then repeat the process above to determine the second, third, ... variable in the returned list.
     
    '''

    def get_fill_size(factor_list, var):
        variables = set()
        for factor in factor_list:
            if factor.get_variable(var.name) is not None:
                variables.update(factor.get_scope())
        return len(variables) - 1

    variables = []
    var_set = set()
    var_set.add(variable_query)
    for factor in factor_list:
        for var in factor.get_scope():
            if var not in var_set:
                var_set.add(var)
                variables.append(var)
    # variables.remove(variable_query)

    ordering = []
    while variables:
        min_size = float('inf')
        min_var = None
        for var in variables:
            fill_size = get_fill_size(factor_list, var)
            if fill_size < min_size:
                min_var = var
                min_size = fill_size

        assert min_var is not None, "Min_fill variable is None."
        ordering.append(min_var)
        variables.remove(min_var)

    return ordering


def ve(bayes_net, var_query, varlist_evidence): 
    '''
    Execute the variable elimination algorithm on the Bayesian network bayes_net
    to compute a distribution over the values of var_query given the 
    evidence provided by varlist_evidence. 

    :param bayes_net: a BN object.
    :param var_query: the query variable. we want to compute a distribution
                     over the values of the query variable.
    :param varlist_evidence: the evidence variables. Each evidence variable has 
                         its evidence set to a value from its domain 
                         using set_evidence.
    :return: a Factor object representing a distribution over the values
             of var_query. that is a list of numbers, one for every value
             in var_query's domain. These numbers sum to 1. The i-th number
             is the probability that var_query is equal to its i-th value given 
             the settings of the evidence variables.

    '''
    org_factors = bayes_net.factors()
    # Restrict evidence variables
    factors = []
    for factor in org_factors:
        for var in varlist_evidence:
            if factor.get_variable(var.name) is not None:
                factor = restrict(factor, var, var.get_evidence())
        factors.append(factor)

    # Eliminate hidden variables
    elim_vars = min_fill_ordering(factors, var_query)
    for var in elim_vars:
        relevant = []
        other = []
        for factor in factors:
            if factor.get_variable(var.name) is not None:
                relevant.append(factor)
            else:
                other.append(factor)
        new_factor = multiply(relevant)
        new_factor = sum_out(new_factor, var)
        factors = other + [new_factor]

    # Multiply remaining factors
    remain_factor = multiply(factors)

    # Normalize
    result = normalize(remain_factor)

    return result


## The order of these domains is consistent with the order of the columns in the data set.
salary_variable_domains = {
"Work": ['Not Working', 'Government', 'Private', 'Self-emp'],
"Education": ['<Gr12', 'HS-Graduate', 'Associate', 'Professional', 'Bachelors', 'Masters', 'Doctorate'],
"Occupation": ['Admin', 'Military', 'Manual Labour', 'Office Labour', 'Service', 'Professional'],
"MaritalStatus": ['Not-Married', 'Married', 'Separated', 'Widowed'],
"Relationship": ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'],
"Race": ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'],
"Gender": ['Male', 'Female'],
"Country": ['North-America', 'South-America', 'Europe', 'Asia', 'Middle-East', 'Carribean'],
"Salary": ['<50K', '>=50K']
}

salary_variable=Variable("Salary", ['<50K', '>=50K'])


def naive_bayes_model(data_file, variable_domains=salary_variable_domains, class_var=salary_variable):
    '''
    NaiveBayesModel returns a BN that is a Naive Bayes model that represents 
    the joint distribution of value assignments to variables in the given dataset.

    Remember a Naive Bayes model assumes P(X1, X2,.... XN, Class) can be represented as 
    P(X1|Class) * P(X2|Class) * .... * P(XN|Class) * P(Class).

    When you generated your Bayes Net, assume that the values in the SALARY column of 
    the dataset are the CLASS that we want to predict.

    Please name the factors as follows. If you don't follow these naming conventions, you will fail our tests.
    - The name of the Salary factor should be called "Salary" without the quotation marks.
    - The name of any other factor should be called "VariableName,Salary" without the quotation marks. 
      For example, the factor for Education should be called "Education,Salary".

    @return a BN that is a Naive Bayes model and which represents the given data set.
    '''
    ### READ IN THE DATA
    input_data = []
    with open(data_file, newline='') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader, None) #skip header row
        for row in reader:
            input_data.append(row)

    def calc_prob(data, var_domains):
        class_name = class_var.name
        class_domain = class_var.domain()
        class_prob = {val: 0 for val in class_domain}
        cond_probs = {(var_name, class_name): {}
                      for var_name in var_domains if var_name != class_name}

        indices = {var_name: headers.index(var_name) for var_name in var_domains}

        for var_name in var_domains:
            if var_name != class_name:
                for val in var_domains[var_name]:
                    for class_val in class_domain:
                        cond_probs[(var_name, class_name)][(val, class_val)] = 0

        for row in data:
            class_val = row[indices[class_name]]
            class_prob[class_val] += 1
            for var_name in var_domains:
                if var_name != class_name:
                    var_val = row[indices[var_name]]
                    cond_probs[(var_name, class_name)][(var_val, class_val)] += 1

        # Calculate probabilities
        for (var_name, class_name), counts in cond_probs.items():
            for (var_val, class_val), count in counts.items():
                prob = count / class_prob[class_val] if count > 0 else 0
                cond_probs[(var_name, class_name)][(var_val, class_val)] = prob

        class_sum = sum(class_prob.values())
        for class_val in class_prob:
            class_prob[class_val] /= class_sum

        return class_prob, cond_probs

    class_name = class_var.name
    variables = [class_var]

    class_prob, cond_probs = calc_prob(input_data, variable_domains)

    class_factor = Factor(class_name, [class_var])
    class_factor.add_values(list(class_prob.items()))
    factors = [class_factor]
    for key, value in cond_probs.items():
        var_name = key[0]
        var = Variable(var_name, variable_domains[var_name])
        variables.append(var)
        factor = Factor(f"{var_name},{class_name}", [var, class_var])
        factor_values = [list(comb) + [val] for comb, val in value.items()]
        factor.add_values(factor_values)
        factors.append(factor)

    return BN("BN", variables, factors)


def explore(bayes_net, question):
    '''    
    Return a probability given a Naive Bayes Model and a question number 1-6. 
    
    The questions are below: 
    1. What percentage of the women in the test data set does our model predict having a salary >= $50K? 
    2. What percentage of the men in the test data set does our model predict having a salary >= $50K? 
    3. What percentage of the women in the test data set satisfies the condition: P(S=">=$50K"|Evidence) is strictly greater than P(S=">=$50K"|Evidence,Gender)?
    4. What percentage of the men in the test data set satisfies the condition: P(S=">=$50K"|Evidence) is strictly greater than P(S=">=$50K"|Evidence,Gender)?
    5. What percentage of the women in the test data set with a predicted salary over $50K (P(Salary=">=$50K"|E) > 0.5) have an actual salary over $50K?
    6. What percentage of the men in the test data set with a predicted salary over $50K (P(Salary=">=$50K"|E) > 0.5) have an actual salary over $50K?

    @return a percentage (between 0 and 100)
    ''' 

    # Read data from test.csv
    input_data = []
    with open('data/adult-test.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader, None) #skip header row
        for row in reader:
            input_data.append(row)

    work = bayes_net.get_variable("Work")
    work_idx = headers.index("Work")

    edu = bayes_net.get_variable("Education")
    edu_idx = headers.index("Education")

    occup = bayes_net.get_variable("Occupation")
    occup_idx = headers.index("Occupation")

    relation = bayes_net.get_variable("Relationship")
    relation_idx = headers.index("Relationship")

    gender = bayes_net.get_variable("Gender")
    gender_idx = headers.index("Gender")

    salary = bayes_net.get_variable("Salary")
    salary_idx = headers.index("Salary")

    if question == 1:
        num_p = 0
        num_high_salary = 0
        for row_data in input_data:
            if row_data[gender_idx] == "Female":
                num_p += 1
                work.set_evidence(row_data[work_idx])
                edu.set_evidence(row_data[edu_idx])
                occup.set_evidence(row_data[occup_idx])
                relation.set_evidence(row_data[relation_idx])
                factor = ve(bayes_net, salary, [work, edu, occup, relation])
                if factor.get_value(['>=50K']) > 0.5:
                    num_high_salary += 1
        percent = num_high_salary / num_p * 100 if num_p != 0 else 0
        return percent

    elif question == 2:
        num_p = 0
        num_high_salary = 0
        for row_data in input_data:
            if row_data[gender_idx] == "Male":
                num_p += 1
                work.set_evidence(row_data[work_idx])
                edu.set_evidence(row_data[edu_idx])
                occup.set_evidence(row_data[occup_idx])
                relation.set_evidence(row_data[relation_idx])
                factor = ve(bayes_net, salary, [work, edu, occup, relation])
                if factor.get_value(['>=50K']) > 0.5:
                    num_high_salary += 1
        percent = num_high_salary / num_p * 100 if num_p != 0 else 0
        return percent
    elif question == 3:
        num_p = 0
        count = 0
        for row_data in input_data:
            num_p += 1
            if row_data[gender_idx] == "Female":
                work.set_evidence(row_data[work_idx])
                edu.set_evidence(row_data[edu_idx])
                occup.set_evidence(row_data[occup_idx])
                relation.set_evidence(row_data[relation_idx])
                gender.set_evidence(row_data[gender_idx])
                fc1 = ve(bayes_net, salary, [work, edu, occup, relation])
                prob1 = fc1.get_value(['>=50K'])
                fc2 = ve(bayes_net, salary, [work, edu, occup, relation, gender])
                prob2 = fc2.get_value(['>=50K'])
                if prob1 > prob2 > 0.5:
                    count += 1
        percent = count / num_p * 100 if num_p != 0 else 0
        return percent
    elif question == 4:
        num_p = 0
        count = 0
        for row_data in input_data:
            num_p += 1
            if row_data[gender_idx] == "Male":
                work.set_evidence(row_data[work_idx])
                edu.set_evidence(row_data[edu_idx])
                occup.set_evidence(row_data[occup_idx])
                relation.set_evidence(row_data[relation_idx])
                gender.set_evidence(row_data[gender_idx])
                fc1 = ve(bayes_net, salary, [work, edu, occup, relation])
                prob1 = fc1.get_value(['>=50K'])
                fc2 = ve(bayes_net, salary, [work, edu, occup, relation, gender])
                prob2 = fc2.get_value(['>=50K'])
                if prob1 > prob2 > 0.5:
                    count += 1
        percent = count / num_p * 100 if num_p != 0 else 0
        return percent
    elif question == 5:
        num_p = 0
        count = 0
        for row_data in input_data:
            if row_data[gender_idx] == "Female":
                work.set_evidence(row_data[work_idx])
                edu.set_evidence(row_data[edu_idx])
                occup.set_evidence(row_data[occup_idx])
                relation.set_evidence(row_data[relation_idx])
                fc = ve(bayes_net, salary, [work, edu, occup, relation])
                prob = fc.get_value(['>=50K'])
                if prob > 0.5:
                    num_p += 1
                    if row_data[salary_idx] == ">=50K":
                        count += 1
        percent = count / num_p * 100 if num_p != 0 else 0
        return percent
    elif question == 6:
        num_p = 0
        count = 0
        for row_data in input_data:
            if row_data[gender_idx] == "Male":
                work.set_evidence(row_data[work_idx])
                edu.set_evidence(row_data[edu_idx])
                occup.set_evidence(row_data[occup_idx])
                relation.set_evidence(row_data[relation_idx])
                fc = ve(bayes_net, salary, [work, edu, occup, relation])
                prob = fc.get_value(['>=50K'])
                if prob > 0.5:
                    num_p += 1
                    if row_data[salary_idx] == ">=50K":
                        count += 1
        percent = count / num_p * 100 if num_p != 0 else 0
        return percent
    else:
        print("Invalid question number.")



# if __name__ == '__main__':
#     # var1 = Variable("A", [1, 2, 3])
#     # var2 = Variable("B", ['a', 'b', 'c'])
#     # var3 = Variable("C", ['x', 'y'])
#     # f1 = Factor("f1", [var1, var2, var3])
#     # f1.add_values([[1, 'a', 'x', 0.5], [1, 'a', 'y', 0.5]])
#     # f2 = Factor("f2", [var1, var2])
#     # f2.add_values([[1, 'a', 0.5], [2, 'b', 0.5]])
#     # f3 = Factor("f3", [var2, var3])
#     # f3.add_values([['a', 'x', 0.1], ['a', 'y', 0.5]])
#     # bn = BN("BN", [var1, var2, var3], [f1, f2, f3])
#     # ve(bn, var1, [var2])
#
#     bn = naive_bayes_model('adult-train.csv')
#     print(explore(bn, 1))
#     print(explore(bn, 2))
#     print(explore(bn, 3))
#     print(explore(bn, 4))
#     print(explore(bn, 5))
#     print(explore(bn, 6))
