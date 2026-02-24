import os
import sys
import signal
import numpy as np

# Session dir is exported by trial.sh. Fallback to 'logs/latest' if run standalone.
SESSION_DIR = os.getenv("SESSION_DIR", "logs/latest")
os.makedirs(SESSION_DIR, exist_ok=True)

interrupted_once = False

def save_session():
    print(f"\n[System] Saving session assets to {SESSION_DIR}...")
    
    # Save Q-Values and total rewards directly to the session dir
    agent.dump_state(os.path.join(SESSION_DIR, "q_values.pkl"))
    np.save(os.path.join(SESSION_DIR, "total_rewards.npy"), total_rewards)
    
    final_score = sum(total_rewards[-20:]) / 20
    print(f"[System] Save complete. Final average score: {final_score:.2f}")

def signal_handler(sig, frame):
    global interrupted_once
    if interrupted_once:
        print("\n[SIGINT] Second interrupt detected. Forcing termination and log cleanup...")
        sys.exit(5)
        
    interrupted_once = True
    print("\n\n--- Execution Interrupted ---")
    print("a) Save accumulated data and exit (Success)")
    print("b) Continue running")
    print("c) Terminate and discard logs")
    
    try:
        choice = input("Select an option (a/b/c): ").lower().strip()
    except EOFError:
        sys.exit(5)

    if choice == 'a':
        save_session()
        sys.exit(0)
    elif choice == 'b':
        print("Resuming execution...\n")
        interrupted_once = False
        return
    elif choice == 'c':
        print("Terminating. Logs will be cleaned up.")
        sys.exit(5)
    else:
        print("Invalid selection. Defaulting to terminate and cleanup.")
        sys.exit(5)

if __name__ == "__main__":
    from runner import GameClient
    from myagent import StudentAgent
    
    client = GameClient()
    agent = StudentAgent()

    signal.signal(signal.SIGINT, signal_handler)

    num_epi = 20_000
    total_rewards = [-400] 
    print("Training started. Press Ctrl+C to manage execution.")
    while len(total_rewards) - 1 < num_epi:
        _, tr = client.run_episode(agent)
        total_rewards.append(tr)

    print("\n[System] Training loop completed organically.")
    save_session()
