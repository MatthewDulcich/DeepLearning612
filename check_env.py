import gymnasium as gym
import flycraft  # This import is what registers the environment

# Get a list of all registered environment IDs
all_envs = list(gym.envs.registry.keys())

print("\n--- Finding FlyCraft Environments ---")
found = False
for env_id in sorted(all_envs):
    if 'flycraft' in env_id.lower():
        print(f"Found registered environment: {env_id}")
        found = True

if not found:
    print("No FlyCraft environments were found in the Gymnasium registry.")
print("-----------------------------------\n")