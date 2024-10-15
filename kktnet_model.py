
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
A_data, b_data, c_data, x_data, lamb_data = load_data('dataset.npz')


# Model
import tensorflow as tf
from tensorflow.keras import layers, models

def create_kkt_net(m, n):
    inputs = layers.Input(shape=(m * n + m + n,))
    x = layers.Dense(256, activation='selu')(inputs)
    x = layers.Dense(256, activation='selu')(x)
    x = layers.Dense(256, activation='selu')(x)

    # Flatten the output for Dense layers
    x = layers.Flatten()(x)

    # Branch for x_opt
    x_opt_branch = layers.Dense(64, activation='selu')(x)
    x_opt = layers.Dense(n)(x_opt_branch)  

    # Branch for lambda_opt
    lambda_opt_branch = layers.Dense(64, activation='selu')(x)
    lambda_opt = layers.Dense(m, activation='relu')(lambda_opt_branch) 

    output = layers.Concatenate()([x_opt, lambda_opt])

    model = models.Model(inputs=inputs, outputs=output)

    return model

def custom_loss(y_true, y_pred, A, b, c, m, n):
    y_pred = tf.convert_to_tensor(y_pred)
    x_opt_pred = y_pred[:, :n]  
    lamb_opt_pred = y_pred[:, n:] 

    A_batch = tf.reshape(A, (-1, m, n))  

    b_batch = tf.reshape(b, (-1, m)) 

    b_expanded = tf.expand_dims(b_batch, axis=-1) 
    x_opt_pred_expanded = tf.expand_dims(x_opt_pred, axis=-1)  
    
    # Primal feasibility loss 
    primal_residual = tf.matmul(A_batch, x_opt_pred_expanded) - b_expanded  
    primal_residual = tf.squeeze(primal_residual, axis=-1)  
    primal_feasibility = tf.reduce_mean(tf.square(tf.nn.relu(primal_residual)), axis=1)  

    # Dual feasibility loss
    lamb_opt_pred_expanded = tf.expand_dims(lamb_opt_pred, axis=-1) 
    dual_feasibility = tf.reduce_mean(tf.square(tf.nn.relu(-lamb_opt_pred_expanded)), axis=1) 

    # Stationarity loss
    c_expanded = tf.expand_dims(c, axis=0) 
    c_expanded = tf.expand_dims(c_expanded, axis=-1) 
    stationarity = tf.reduce_mean(tf.square(c_expanded + tf.matmul(A_batch, lamb_opt_pred_expanded, transpose_a=True)), axis=1)  
    
    # Complementary slackness loss
    primal_residual_expanded = tf.expand_dims(primal_residual, axis=-1)  
    complementary_slackness = tf.reduce_mean(tf.square(lamb_opt_pred_expanded * primal_residual_expanded), axis=1)  

    kkt_loss = 0.1 * tf.reduce_mean(primal_feasibility) + 0.1 * tf.reduce_mean(dual_feasibility) + 0.6 * tf.reduce_mean(stationarity) + 0.2 * tf.reduce_mean(complementary_slackness)

    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))

    return kkt_loss + mse_loss 

indices = np.random.permutation(num_samples)
split_index = int(0.75 * num_samples)

# Split indices into training and validation sets
train_indices = indices[:split_index]
val_indices = indices[split_index:]

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

train_A_data_tf = tf.convert_to_tensor(train_A_data, dtype=tf.float32)
train_b_data_tf = tf.convert_to_tensor(train_b_data, dtype=tf.float32)
train_c_data_tf = tf.convert_to_tensor(train_c_data, dtype=tf.float32)
train_x_data_tf = tf.convert_to_tensor(train_x_data, dtype=tf.float32)
train_lamb_data_tf = tf.cast(tf.convert_to_tensor(train_lamb_data), dtype=tf.float32)

# Convert the validation data to tensors
val_A_data_tf = tf.convert_to_tensor(val_A_data, dtype=tf.float32)
val_b_data_tf = tf.convert_to_tensor(val_b_data, dtype=tf.float32)
val_c_data_tf = tf.convert_to_tensor(val_c_data, dtype=tf.float32)
val_x_data_tf = tf.convert_to_tensor(val_x_data, dtype=tf.float32)
val_lamb_data_tf = tf.cast(tf.convert_to_tensor(val_lamb_data), dtype=tf.float32)

print(f"Size of train_A_data: {train_A_data_tf.shape}")
print(f"Size of train_b_data: {train_b_data_tf.shape}")
print(f"Size of train_c_data: {train_c_data_tf.shape}")
print(f"Size of train_x_data: {train_x_data_tf.shape}")
print(f"Size of train_lamb_data: {train_lamb_data_tf.shape}")

train_A_flatten = tf.reshape(train_A_data_tf, (split_index, -1)) 
train_input_data = tf.concat([train_A_flatten, train_b_data_tf, train_c_data_tf], axis=1)  

val_A_flatten = tf.reshape(val_A_data_tf, (num_samples - split_index, -1))  
val_input_data = tf.concat([val_A_flatten, val_b_data_tf, val_c_data_tf], axis=1)  

train_output_data = tf.concat([train_x_data_tf, train_lamb_data_tf], axis=1)  
val_output_data = tf.concat([val_x_data_tf, val_lamb_data_tf], axis=1)  

# Create TensorFlow datasets for training and validation
batch_size = 256
train_dataset = tf.data.Dataset.from_tensor_slices((train_input_data, train_output_data, train_A_data_tf, train_b_data_tf, train_c_data_tf)).batch(batch_size, drop_remainder=True)
val_dataset = tf.data.Dataset.from_tensor_slices((val_input_data, val_output_data, val_A_data_tf, val_b_data_tf, val_c_data_tf)).batch(batch_size, drop_remainder=True)

model = create_kkt_net(m, n)
optimizer = tf.keras.optimizers.Adam()

# Training step
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

epochs = 2000
save_interval = 1000

train_losses = []
val_losses = []

# Training loop
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

    print(f'Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')

    if (epoch + 1) % save_interval == 0:
        model.save(f'KKTNetmodel_checkpoint_{epoch + 1:04d}.keras')
        print('Model saved...')

np.save('train_loss.npy', np.array(train_losses))
np.save('val_loss.npy', np.array(val_losses))
