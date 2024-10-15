
# Parameters
num_samples = 2**10  
m = 2  # Number of inequality constraints
n = 2  # Number of variables


import numpy as np

# Function to load data from a local file
def load_data(filename):
    data = np.load(filename)
    A_data = data['A']
    b_data = data['b']
    c_data = data['c']
    x_data = data['x']
    lamb_data = data['lamb']
    return A_data, b_data, c_data, x_data, lamb_data

# Load the dataset
A_data, b_data, c_data, x_data, lamb_data = load_data('/content/drive/My Drive/btp/dataset.npz')


# Model
import tensorflow as tf
from tensorflow.keras import layers, models

def create_kkt_net(m, n):
    # inputs = layers.Input(shape=(m * n + m + n, 1))
    # x = layers.Conv1D(64, kernel_size=3, activation='selu', padding='same')(inputs)
    # x = layers.Conv1D(64, kernel_size=3, activation='selu', padding='same')(x)
    # x = layers.Conv1D(64, kernel_size=3, activation='selu', padding='same')(x)
    inputs = layers.Input(shape=(m * n + m + n,))
    x = layers.Dense(256, activation='selu')(inputs)
    x = layers.Dense(256, activation='selu')(x)
    x = layers.Dense(256, activation='selu')(x)

    # Flatten the output for Dense layers
    x = layers.Flatten()(x)

    # Branch for x_opt
    # x_opt_branch = layers.Conv1D(64, kernel_size=3, activation='selu', padding='same')(x)  # Additional layer for x_opt
    x_opt_branch = layers.Dense(64, activation='selu')(x)
    # x_opt_branch = layers.Dense(64, activation='selu')(x_opt_branch)
    x_opt = layers.Dense(n)(x_opt_branch)  # Output x_opt

    # Branch for lambda_opt
    # lambda_opt_branch = layers.Conv1D(64, kernel_size=3, activation='selu', padding='same')(x)  # Additional layer for lambda_opt
    lambda_opt_branch = layers.Dense(64, activation='selu')(x)
    # lambda_opt_branch = layers.Dense(64, activation='selu')(lambda_opt_branch)
    lambda_opt = layers.Dense(m, activation='relu')(lambda_opt_branch)  # Output lambda_opt

    # Concatenate outputs
    output = layers.Concatenate()([x_opt, lambda_opt])

    # Define a custom model
    model = models.Model(inputs=inputs, outputs=output)

    return model

# def create_kkt_net(m, n, num_heads=4, ff_dim=128):
#     # Inputs
#     inputs = layers.Input(shape=(m * n + m + n, 1))

#     # Flatten input data
#     flattened_inputs = layers.Flatten()(inputs)

#     # Dense layer to project to transformer input size
#     x = layers.Dense(ff_dim, activation='selu')(flattened_inputs)

#     # Reshape to match transformer input shape
#     x = layers.Reshape((8, ff_dim // 8))(x)  # Reshaped to ensure compatibility for MultiHeadAttention

#     # Transformer Encoder layer
#     transformer_block = layers.MultiHeadAttention(num_heads=num_heads, key_dim=ff_dim // 8)(x, x)
#     transformer_block = layers.LayerNormalization(epsilon=1e-6)(transformer_block)
#     transformer_block = layers.Dropout(0.1)(transformer_block)

#     # Feed-forward network within transformer
#     ff_block = layers.Dense(ff_dim // 8, activation='selu')(transformer_block)  # Adjusted to match shape
#     ff_block = layers.Dense(ff_dim // 8)(ff_block)  # Adjusted to match shape
#     ff_block = layers.LayerNormalization(epsilon=1e-6)(ff_block)
#     ff_block = layers.Dropout(0.1)(ff_block)

#     # Now both transformer_block and ff_block have the same shape, so we can add them
#     transformer_output = layers.Add()([transformer_block, ff_block])

#     # Flatten the output for Dense layers
#     x = layers.Flatten()(transformer_output)

#     # Branch for x_opt
#     x_opt_branch = layers.Dense(64, activation='selu')(x)
#     x_opt_branch = layers.Dense(64, activation='selu')(x_opt_branch)
#     x_opt = layers.Dense(n)(x_opt_branch)  # Output x_opt

#     # Branch for lambda_opt
#     lambda_opt_branch = layers.Dense(64, activation='selu')(x)
#     lambda_opt_branch = layers.Dense(64, activation='selu')(lambda_opt_branch)
#     lambda_opt = layers.Dense(m, activation='relu')(lambda_opt_branch)  # Output lambda_opt

#     # Concatenate outputs
#     output = layers.Concatenate()([x_opt, lambda_opt])

#     # Define a custom model
#     model = models.Model(inputs=inputs, outputs=output)

#     return model

def custom_loss(y_true, y_pred, A, b, c, m, n):
    y_pred = tf.convert_to_tensor(y_pred)
    x_opt_pred = y_pred[:, :n]  # shape: (batch_size, n)
    lamb_opt_pred = y_pred[:, n:]  # shape: (batch_size, m)

    # Ensure A has shape (batch_size, m, n)
    A_batch = tf.reshape(A, (-1, m, n))  # shape: (batch_size, m, n)

    # Ensure b has shape (batch_size, m)
    b_batch = tf.reshape(b, (-1, m))  # shape: (batch_size, m)

    # Expand dimensions for broadcasting
    b_expanded = tf.expand_dims(b_batch, axis=-1)  # shape: (batch_size, m, 1)
    x_opt_pred_expanded = tf.expand_dims(x_opt_pred, axis=-1)  # shape: (batch_size, n, 1)

    # Primal feasibility loss using squared error
    primal_residual = tf.matmul(A_batch, x_opt_pred_expanded) - b_expanded  # shape: (batch_size, m, 1)
    primal_residual = tf.squeeze(primal_residual, axis=-1)  # shape: (batch_size, m)
    #primal_feasibility = tf.reduce_mean(tf.square(primal_residual), axis=1)  # mean squared error over batch
    primal_feasibility = tf.reduce_mean(tf.square(tf.nn.relu(primal_residual)), axis=1)  # mean squared error over batch
    # primal_feasibility = tf.reduce_mean(tf.exp(primal_residual), axis=1)

    # Dual feasibility loss
    lamb_opt_pred_expanded = tf.expand_dims(lamb_opt_pred, axis=-1)  # shape: (batch_size, m, 1)
    dual_feasibility = tf.reduce_mean(tf.square(tf.nn.relu(-lamb_opt_pred_expanded)), axis=1)  # mean squared error over batch
    # dual_feasibility = tf.reduce_mean(tf.exp(-lamb_opt_pred_expanded), axis=1)

    # Stationarity loss
    c_expanded = tf.expand_dims(c, axis=0)  # shape: (batch_size, n)
    c_expanded = tf.expand_dims(c_expanded, axis=-1)  # shape: (batch_size, n, 1)
    stationarity = tf.reduce_mean(tf.square(c_expanded + tf.matmul(A_batch, lamb_opt_pred_expanded, transpose_a=True)), axis=1)  # mean squared error over batch
    # stationarity = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true=tf.zeros_like(c_expanded),y_pred=tf.sigmoid(c_expanded + tf.matmul(A_batch, lamb_opt_pred_expanded, transpose_a=True))),axis=1)

    # Complementary slackness loss
    primal_residual_expanded = tf.expand_dims(primal_residual, axis=-1)  # shape: (batch_size, m, 1)
    complementary_slackness = tf.reduce_mean(tf.square(lamb_opt_pred_expanded * primal_residual_expanded), axis=1)  # mean squared error over batch

    kkt_loss = 0.1 * tf.reduce_mean(primal_feasibility) + 0.1 * tf.reduce_mean(dual_feasibility) + 0.6 * tf.reduce_mean(stationarity) + 0.2 * tf.reduce_mean(complementary_slackness)

    # MSE Loss
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))

    return kkt_loss + mse_loss #

# Split A_data, b_data, c_data, x_data, lamb_data
indices = np.random.permutation(num_samples)

# Determine the split index
split_index = int(0.75 * num_samples)

# Split indices into training and validation sets
train_indices = indices[:split_index]
val_indices = indices[split_index:]

# Split the data based on the random indices
train_A_data = A_data[train_indices]
train_b_data = b_data[train_indices]
train_c_data = c_data[train_indices]
train_x_data = x_data[train_indices]
train_lamb_data = tf.reshape(lamb_data[train_indices], (split_index, 2))

val_A_data = A_data[val_indices]
val_b_data = b_data[val_indices]
val_c_data = c_data[val_indices]
val_x_data = x_data[val_indices]
val_lamb_data = lamb_data[val_indices]

# Convert the training data to tensors
train_A_data_tf = tf.convert_to_tensor(train_A_data, dtype=tf.float32)
train_b_data_tf = tf.convert_to_tensor(train_b_data, dtype=tf.float32)
train_c_data_tf = tf.convert_to_tensor(train_c_data, dtype=tf.float32)
train_x_data_tf = tf.convert_to_tensor(train_x_data, dtype=tf.float32)
# train_lamb_data = train_lamb_data.astype(np.float32)
train_lamb_data_tf = tf.cast(tf.convert_to_tensor(train_lamb_data), dtype=tf.float32)

# Convert the validation data to tensors
val_A_data_tf = tf.convert_to_tensor(val_A_data, dtype=tf.float32)
val_b_data_tf = tf.convert_to_tensor(val_b_data, dtype=tf.float32)
val_c_data_tf = tf.convert_to_tensor(val_c_data, dtype=tf.float32)
val_x_data_tf = tf.convert_to_tensor(val_x_data, dtype=tf.float32)
# val_lamb_data = val_lamb_data.astype(np.float32)
val_lamb_data_tf = tf.cast(tf.convert_to_tensor(val_lamb_data), dtype=tf.float32)

print(f"Size of train_A_data: {train_A_data_tf.shape}")
print(f"Size of train_b_data: {train_b_data_tf.shape}")
print(f"Size of train_c_data: {train_c_data_tf.shape}")
print(f"Size of train_x_data: {train_x_data_tf.shape}")
print(f"Size of train_lamb_data: {train_lamb_data_tf.shape}")

# Flatten and concatenate A, b, and c for model input
train_A_flatten = tf.reshape(train_A_data_tf, (split_index, -1))  # Shape: (split_index, 4)

# Concatenate flattened A_data with b_data and c_data along axis=1
train_input_data = tf.concat([train_A_flatten, train_b_data_tf, train_c_data_tf], axis=1)  # Final shape: (split_index, 8)

# Similarly for the validation data
val_A_flatten = tf.reshape(val_A_data_tf, (num_samples - split_index, -1))  # Shape: (num_samples - split_index, 4)
val_input_data = tf.concat([val_A_flatten, val_b_data_tf, val_c_data_tf], axis=1)  # Final shape: (num_samples - split_index, 8)

# Concatenate x_data and lamb_data along axis=1 for model output
train_output_data = tf.concat([train_x_data_tf, train_lamb_data_tf], axis=1)  # Final shape: (split_index, 4)
val_output_data = tf.concat([val_x_data_tf, val_lamb_data_tf], axis=1)  # Final shape: (num_samples - split_index, 4)

# Create TensorFlow datasets for training and validation
batch_size = 256
train_dataset = tf.data.Dataset.from_tensor_slices((train_input_data, train_output_data, train_A_data_tf, train_b_data_tf, train_c_data_tf)).batch(batch_size, drop_remainder=True)
val_dataset = tf.data.Dataset.from_tensor_slices((val_input_data, val_output_data, val_A_data_tf, val_b_data_tf, val_c_data_tf)).batch(batch_size, drop_remainder=True)


# Instantiate the model
model = create_kkt_net(m, n)
optimizer = tf.keras.optimizers.Adam()

# Custom training step
def train_step(inputs, outputs, A, b, c):
    with tf.GradientTape() as tape:
        y_pred = model(inputs, training=True)
        loss = custom_loss(outputs, y_pred, A, b, c, m, n)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Validation step
def val_step(inputs, outputs, A, b, c):
    y_pred = model(inputs, training=False)
    loss = custom_loss(outputs, y_pred, A, b, c, m, n)
    return loss

# Training loop
epochs = 2000
save_interval = 1000

train_losses = []
val_losses = []

for epoch in range(epochs):
    # Training phase
    epoch_train_loss = 0
    num_train_batches = 0
    for batch_inputs, batch_outputs, batch_A, batch_b, batch_c in train_dataset:
        loss = train_step(batch_inputs, batch_outputs, batch_A, batch_b, batch_c)
        epoch_train_loss += loss.numpy()
        num_train_batches += 1

    avg_train_loss = epoch_train_loss / num_train_batches
    train_losses.append(avg_train_loss)

    # Validation phase
    epoch_val_loss = 0
    num_val_batches = 0
    for batch_inputs, batch_outputs, batch_A, batch_b, batch_c in val_dataset:
        loss = val_step(batch_inputs, batch_outputs, batch_A, batch_b, batch_c)
        epoch_val_loss += loss.numpy()
        num_val_batches += 1

    avg_val_loss = epoch_val_loss / num_val_batches
    val_losses.append(avg_val_loss)

    # Print losses
    print(f'Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')


    # Save the model every save_interval epochs
    if (epoch + 1) % save_interval == 0:
        model.save(f'/content/drive/My Drive/btp/KKTNetmodelBranchedOut_checkpoint_{epoch + 1:04d}.keras')
        print('Model saved...')

# Save the loss arrays to files after training
np.save('/content/drive/My Drive/btp/KKTNet_train_loss.npy', np.array(train_losses))
np.save('/content/drive/My Drive/btp/KKTNet_val_loss.npy', np.array(val_losses))
