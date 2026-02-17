# Idea
I want to try using Iterative Policy Improvement for this assignment. However,
I am lacking information on the following:
    - Environment dynamic (p)
    - State size (I think its finite)

I believe in class Dr. Ishigaki said that rewards are deterministic.

I think I can also use one of the three methods we used for exploration
in order to sample the state space.

But if initially the state space is unknown, how do I initialize the
environment dynamic?

Another idea is that I could use epsilon-greedy, and gradually anneal
the epsilon over the number of episodes. This would allow me to sample more
of the state space before settling on a strategy.

How should I represent state, action, and transitions? Can I apply that same
upper-confidence bound for the transition probability estimates?

More importantly, how should I keep track of the states and actions?
Multi-leveled map?

state-action-space = {
    (0,1): {
        0: {
            (0,2): 1, 
            (1,0): 2 
        },
        1: {
            (0,3): 1, 
            (1,0): 3
        }
    }
}
