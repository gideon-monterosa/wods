from feature_interaction.synthetic_data_generator import SyntheticDataGenerator

dataGenerator = SyntheticDataGenerator()

X, y = dataGenerator.generate_complex_synthetic_dataset(1000)
print(X)
print(y)
