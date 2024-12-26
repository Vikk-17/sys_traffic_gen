import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def load_data(filepath):
    data = pd.read_csv(filepath)
    data.columns = data.columns.str.strip()  # Removing whitespaces
    data = data.dropna()
    return data


def preprocess_data(data):
    labels = data["Label"].copy()
    features = data.drop(columns=["Label"])

    # Encode labels to binary (0 for normal, 1 for attack)
    labels = (labels != "BENIGN").astype(int)

    # Replace infinite values with NaN
    features = features.replace([np.inf, -np.inf], np.nan)

    # Fill NaN values using suitable strategy (e.g., mean or median)
    for col in features.columns:
        features[col] = features[col].fillna(features[col].mean())

    # Normalize features
    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)

    return features, labels, scaler


# Generator
class Generator(tf.keras.Model):
    def __init__(self, noise_dim, feature_dim):
        super(Generator, self).__init__()
        # Creating layers
        self.dense1 = tf.keras.layers.Dense(units=128, activation="relu")
        self.dense2 = tf.keras.layers.Dense(units=256, activation="relu")
        self.dense3 = tf.keras.layers.Dense(feature_dim, activation="tanh")

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.dense3(x)


# Discriminator
class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.dense1 = tf.keras.layers.Dense(units=256, activation="relu")
        self.dense2 = tf.keras.layers.Dense(units=128, activation="relu")
        self.dense3 = tf.keras.layers.Dense(units=1, activation="sigmoid")

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.dense3(x)


# Train the Model
def train_gen(generator, discriminator, features, epochs, batch_size, noise_dim):
    cross_entropy = tf.keras.losses.BinaryCrossentropy()

    gen_optimizer = tf.keras.optimizers.Adam(1e-4)
    disc_optimizer = tf.keras.optimizers.Adam(1e-4)

    @tf.function
    def train_step(real_data):
        noise = tf.random.normal([batch_size, noise_dim])
        generated_data = generator(noise)

        with tf.GradientTape() as disc_tape:
            real_output = discriminator(real_data)
            fake_output = discriminator(generated_data)
            disc_loss = cross_entropy(
                tf.ones_like(real_output), real_output
            ) + cross_entropy(tf.zeros_like(fake_output), fake_output)

        gradients_of_discriminator = disc_tape.gradient(
            disc_loss, discriminator.trainable_variables
        )
        disc_optimizer.apply_gradients(
            zip(gradients_of_discriminator, discriminator.trainable_variables)
        )

        with tf.GradientTape() as gen_tape:
            generated_data = generator(noise)
            fake_output = discriminator(generated_data)
            gen_loss = cross_entropy(tf.ones_like(fake_output), fake_output)

        gradients_of_generator = gen_tape.gradient(
            gen_loss, generator.trainable_variables
        )
        gen_optimizer.apply_gradients(
            zip(gradients_of_generator, generator.trainable_variables)
        )

        return gen_loss, disc_loss

    for epoch in range(epochs):
        for i in range(0, len(features), batch_size):
            batch_data = features[i : i + batch_size]
            gen_loss, disc_loss = train_step(batch_data)
        print(
            f"Epoch {epoch + 1}, Generator loss: {gen_loss.numpy()}, Discrimintor loss: {disc_loss.numpy()}"
        )


# Generate synthetic traffic
def generate_synthetic_traffic(generator, num_samples, noise_dim):
    noise = tf.random.uniform([num_samples, noise_dim], maxval=1, minval=-1)
    # set training false for inference
    synthetic_data = generator(noise, training=False)
    return synthetic_data.numpy()


def main():
    data = load_data("/content/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
    features, labels, scaler = preprocess_data(data)

    noise_dim = 100
    feature_dim = features.shape[1]

    generator = Generator(noise_dim, feature_dim)
    discriminator = Discriminator()

    train_gen(
        generator,
        discriminator,
        features,
        epochs=100,
        batch_size=32,
        noise_dim=noise_dim,
    )

    # Generate synthetic traffic after training
    num_samples = 1000  # Number of synthetic samples to generate
    synthetic_traffic = generate_synthetic_traffic(generator, num_samples, noise_dim)

    # Inverse transform to original scale
    synthetic_traffic = scaler.inverse_transform(synthetic_traffic)

    # Save to a csv file
    synthetic_df = pd.DataFrame(
        synthetic_traffic, columns=data.drop(columns=["Label"]).columns
    )
    synthetic_df.to_csv("synthetic_traffic_ddos.csv", index=False)
    print("Synthetic traffic saved to 'synthetic_traffic_ddos.csv'")


if __name__ == "__main__":
    main()
