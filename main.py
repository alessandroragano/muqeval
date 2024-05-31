#! /home/alergn/virtualenvs/torchgpu/bin/python3.9
import click
import sys
import yaml

@click.command()
@click.option('--config_file', type=str)
def training(config_file):
    
    # Open config file
    with open(config_file) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # Import script (this changes based on which config file arg is passed by the user)
    module_name = config['exp_script']
    __import__(module_name)

    # Make imported module variables available
    imported_module = sys.modules[module_name]

    # Create new training object
    exp_obj = imported_module.Experiment(config)

    # Run experiment
    if (config['exp_script'] == 'src.exp1') | (config['exp_script'] == 'src.exp2') | (config['exp_script'] == 'src.exp3'):
        exp_obj.samples_loop()

if __name__ == '__main__':
    training()  