"""Start, resume or check a training configuration."""
def main(argv=None):
    from robot_control.runtime.train import main as train
    return train(argv)
