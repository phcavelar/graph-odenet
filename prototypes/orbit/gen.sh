echo "Preparing folders"
rm -r data
mkdir data
mkdir data/3
mkdir data/6
mkdir data/12
rm -r dataset
mkdir dataset
mkdir dataset/3
mkdir dataset/6
mkdir dataset/12
echo "Generating Training Scenes"

max_timesteps=1000
num_scenes=1000
half_scenes=500

# Training scenes
python3 run_simulation.py --save_data --start_at=0 --num_scenes=$num_scenes --max_timesteps=$max_timesteps --num_of_bodies=6 --orbit_type=elliptical
python3 run_simulation.py --save_data --start_at=$num_scenes --num_scenes=$num_scenes --max_timesteps=$num_scenes --num_of_bodies=6 --orbit_type=random
python3 prepare_dataset.py --num_of_bodies=6 --train_pct=0.5 --test_pct=0.5 --val_pct=0 --max_timesteps=$max_timesteps

echo "Generating Test Scenes"
# Test scenes
python3 run_simulation.py --save_data --start_at=0 --num_scenes=$half_scenes --max_timesteps=$max_timesteps --num_of_bodies=3 --orbit_type=elliptical
python3 run_simulation.py --save_data --start_at=$half_scenes --num_scenes=$half_scenes --max_timesteps=$max_timesteps --num_of_bodies=3 --orbit_type=random
python3 prepare_dataset.py --num_of_bodies=3 --train_pct=0 --test_pct=1 --val_pct=0 --max_timesteps=$max_timesteps
python3 run_simulation.py --save_data --start_at=0 --num_scenes=$half_scenes --max_timesteps=$max_timesteps --num_of_bodies=12 --orbit_type=elliptical
python3 run_simulation.py --save_data --start_at=$half_scenes --num_scenes=$half_scenes --max_timesteps=$max_timesteps --num_of_bodies=12 --orbit_type=random
python3 prepare_dataset.py --num_of_bodies=12 --train_pct=0 --test_pct=1 --val_pct=0 --max_timesteps=$max_timesteps

