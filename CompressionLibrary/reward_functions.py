import tensorflow as tf


def reward_1minus1(stats: dict) -> float:
   return 1 - (stats['weights_after']/stats['weights_before']) + 2 * (stats['accuracy_after'] - stats['accuracy_before'])

def reward_original(stats: dict) -> float:
   return 1 - (stats['weights_after']/stats['weights_before']) +  (stats['accuracy_after'] - 0.9 * stats['accuracy_before'])