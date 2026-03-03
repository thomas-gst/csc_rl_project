import asyncio
from poke_env import RandomPlayer
from poke_env.data import GenData

random_player = RandomPlayer()
second_player = RandomPlayer()


asyncio.run(random_player.battle_against(second_player, n_battles=1))

print(
    f"Player {random_player.username} won {random_player.n_won_battles} out of {random_player.n_finished_battles} played"
)
print(
    f"Player {second_player.username} won {second_player.n_won_battles} out of {second_player.n_finished_battles} played"
)

for battle_tag, battle in random_player.battles.items():
    print(battle_tag, battle.won)
