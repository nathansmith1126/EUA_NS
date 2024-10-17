from automata.fa.dfa import DFA
import numpy as np 

def create_dfa(n, m):
    # Define DFA states
    states = {f'q{i}' for i in range(n + m + 1)}  # States q0 to q(n+m)
    
    # Initial state
    initial_state = 'q0'
    
    # Accepting state (after n 'A's and m 'B's)
    accepting_states = {f'q{n+m}'}
    
    # Transition function
    transitions = {}
    
    # Transitions for 'A's (first n transitions)
    for i in range(n):
        transitions[(f'q{i}', 'A')] = f'q{i+1}'
    
    # Transitions for 'B's (next m transitions)
    for i in range(n, n + m):
        transitions[(f'q{i}', 'B')] = f'q{i+1}'
    
    # Create DFA
    dfa = DFA(
        states=states,
        input_symbols={'A', 'B'},
        transitions=transitions,
        initial_state=initial_state,
        final_states=accepting_states
    )
    
    return dfa

# Example: Create DFA that accepts 3 'A's followed by 2 'B's
n = 3
m = 2
dfa = create_dfa(n, m)

# Test DFA
test_string = "AAABB"  # This should be accepted
print(dfa.accepts_input(test_string))  # Output: True

test_string = "AABBB"  # This should be rejected
print(dfa.accepts_input(test_string))  # Output: False
