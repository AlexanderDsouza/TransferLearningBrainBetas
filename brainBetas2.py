import pandas as pd
import numpy as np

from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE  # Import SMOTE
from sklearn.model_selection import train_test_split



from tensorflow import keras
from tensorflow.keras import models, layers
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import layers, Sequential
from tensorflow.keras.optimizers import Adam
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.metrics import Accuracy



def load_data(file_path):
    """Load datasets from the specified Excel file."""
    outcome_df = pd.read_excel(file_path, sheet_name='outcomes')
    scanner_ids = pd.read_excel(file_path, sheet_name='scannerid')
    subset_scanner_ids = scanner_ids[scanner_ids['EP1or2'] == 1]
    usable_outcomes = outcome_df[outcome_df['SID'].isin(subset_scanner_ids['SID'])]
    fmrifeatures_df = pd.read_excel(file_path, sheet_name='rawfmrifeatures') 
    fmrifeatures_df = fmrifeatures_df[fmrifeatures_df['SID'].isin(subset_scanner_ids['SID'])]

    clinfeatues_df = pd.read_excel(file_path, sheet_name = 'clinfeatures')
    clinfeatues_df = clinfeatues_df[clinfeatues_df['SID'].isin(subset_scanner_ids['SID'])]

    usable_outcomes = usable_outcomes.merge(clinfeatues_df,on='SID', how='inner')

    return usable_outcomes, fmrifeatures_df

def load_data2(file_path):
    """Load datasets from the specified Excel file."""
    outcome_df = pd.read_excel(file_path, sheet_name='outcomes')
    scanner_ids = pd.read_excel(file_path, sheet_name='scannerid')
    subset_scanner_ids = scanner_ids[scanner_ids['EP1or2'] == 2]
    usable_outcomes = outcome_df[outcome_df['SID'].isin(subset_scanner_ids['SID'])]
    fmrifeatures_df = pd.read_excel(file_path, sheet_name='rawfmrifeatures') 
    fmrifeatures_df = fmrifeatures_df[fmrifeatures_df['SID'].isin(subset_scanner_ids['SID'])]

    clinfeatues_df = pd.read_excel(file_path, sheet_name = 'clinfeatures')
    clinfeatues_df = clinfeatues_df[clinfeatues_df['SID'].isin(subset_scanner_ids['SID'])]

    usable_outcomes = usable_outcomes.merge(clinfeatues_df,on='SID', how='inner')

    return usable_outcomes, fmrifeatures_df

def calculate_cue_differences(fmrifeatures_df):
    """Calculate CueB - CueA differences and return as a DataFrame."""
    new_columns = {'SID': fmrifeatures_df['SID']}
    for col in fmrifeatures_df.columns:
        if col.startswith('CueA_'):
            variable_part = col[5:]
            cueb_col = f'CueB_{variable_part}'
            if cueb_col in fmrifeatures_df.columns:
                new_col_name = f'CueBMinusA_{variable_part}'
                new_columns[new_col_name] = fmrifeatures_df[cueb_col] - fmrifeatures_df[col]
    return pd.DataFrame(new_columns)

def merge_data(usable_outcomes, cueB_minus_cueA):
    """Merge usable outcomes with CueB - CueA differences and remove identical rows."""
    merged_df_with_SID = usable_outcomes.merge(cueB_minus_cueA, on='SID', how='inner')
    merged_df_with_SID = merged_df_with_SID.drop_duplicates()
    return merged_df_with_SID.drop(columns=['Chg_BPRS', 'SID'])

def merge_demographics_data(usable_outcomes, cueB_minus_cueA, file_path):
    """merging the demographics sheet"""
    demographic_df = pd.read_excel(file_path, sheet_name='demographics')
    demographic_df = demographic_df[demographic_df['SID'].isin(usable_outcomes['SID'])]
    merged_demographic_df_with_SID = usable_outcomes.merge(demographic_df, on='SID', how='inner')
    merged_demographic_df_with_SID = usable_outcomes.merge(cueB_minus_cueA, on='SID', how='inner')
    return merged_demographic_df_with_SID.drop(columns=['Chg_BPRS','SID'])

def prepare_data(merged_df):
    """Prepare features and labels from the merged DataFrame."""
    X = merged_df.drop(columns=['Imp20PercentBPRS']).values
    y = merged_df['Imp20PercentBPRS'].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y

def prepare_data_with_pca(merged_df, n_components=10):
    """Prepare features and labels from the merged DataFrame using PCA for dimensionality reduction."""
    X = merged_df.drop(columns=['Imp20PercentBPRS']).values
    y = merged_df['Imp20PercentBPRS'].values
    
    # Standardizing the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Applying PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    
    return X_pca, y



def create_model(input_shape, learning_rate=0.001, dropout_rate=0.3):
    model = Sequential([
        layers.Input(shape=(input_shape,)),
        layers.Dense(128),
        layers.LeakyReLU(negative_slope=0.01),
        layers.Dropout(dropout_rate),

        layers.Dense(64),
        layers.LeakyReLU(negative_slope=0.01),
        layers.Dropout(dropout_rate),

        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    return model



def evaluate_model(X, y):
    """Perform Leave-One-Out Cross-Validation and evaluate the model with confusion matrix."""
    loo = LeaveOneOut()
    accuracies = []
    predicted_probs = []
    y_true_all = []
    y_pred_all = []

    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Apply SMOTE for balancing
        smote = SMOTE()
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        
        # Initialize model with current data shape
        model = create_model(input_shape=X_train.shape[1])
        
        # Compute class weights
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = {i: class_weights[i] for i in np.unique(y_train)}
        
        # Train model with resampled data and class weights
        model.fit(X_train_resampled, y_train_resampled, epochs=10, batch_size=32, class_weight=class_weight_dict, verbose=0)
        
        # Evaluate model - get only the accuracy
        results = model.evaluate(X_test, y_test, verbose=0)
        accuracy = results[1]  # Extract accuracy assuming it's the second value returned
        accuracies.append(accuracy)
        
        # Predict and store probabilities for AUC calculation
        y_pred_prob = model.predict(X_test).flatten()
        y_pred = (y_pred_prob > 0.5).astype(int)  # Convert probabilities to binary predictions
        predicted_probs.append(y_pred_prob[0])
        y_true_all.append(y_test[0])
        y_pred_all.append(y_pred[0])

    # Calculate metrics
    average_accuracy = np.mean(accuracies)
    auc = roc_auc_score(y_true_all, predicted_probs)
    conf_matrix = confusion_matrix(y_true_all, y_pred_all)
    
    return average_accuracy, auc, conf_matrix


def build_encoder(fmrifeatures_df, latent_dim):
    encoder = models.Sequential([
        layers.InputLayer(input_shape=(fmrifeatures_df.shape[1],)),  # Input size depends on your features
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(latent_dim, activation='relu')  # Latent dimension
    ])
    return encoder

def build_decoder(fmrifeatures_df, latent_dim):
    decoder = models.Sequential([
        layers.InputLayer(input_shape=(latent_dim,)),  # Latent space input
        layers.Dense(64, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(fmrifeatures_df.shape[1], activation='sigmoid')  # Output should match feature space
    ])
    return decoder

def train_encoder_decoder(encoder, decoder, source_data, target_data, epochs=10, batch_size=32, test_size=0.2):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(source_data, target_data, test_size=test_size, random_state=42)

    # Create the model where we encode and then decode
    inputs = layers.Input(shape=(X_train.shape[1],))
    latent_representation = encoder(inputs)
    reconstructed_output = decoder(latent_representation)
    model = models.Model(inputs=inputs, outputs=reconstructed_output)
    
    # Compile the model with a reconstruction loss (e.g., Mean Squared Error)
    model.compile(optimizer='adam', loss='mse')
    
    # Train the model
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    
    # Evaluate the model on the test set
    test_loss = model.evaluate(X_test, y_test, verbose=1)
    
    print(f'Test MSE: {test_loss}')
    
    return model, encoder, decoder

def save_decoder(decoder, filepath):
    decoder.save(filepath)

def create_new_encoder(fmrifeatures_df_EP2, decoder, latent_dim):
    # Freeze the decoder's weights to keep them fixed
    decoder.trainable = False
    # Create a new encoder model that uses the saved decoder
    encoder = build_encoder(fmrifeatures_df_EP2, latent_dim)
    inputs = layers.Input(shape=(fmrifeatures_df_EP2.shape[1],))  # EP2 features
    latent_representation = encoder(inputs)
    reconstructed_output = decoder(latent_representation)
    
    model = models.Model(inputs=inputs, outputs=reconstructed_output)
    return model

def fine_tune_encoder(new_encoder, ep2_data, epochs=10, test_size=0.2, batch_size=32):
    # Split the data into training and testing sets
    X_train, X_test = train_test_split(ep2_data, test_size=test_size, random_state=42)

    # Train the encoder on the training data
    new_encoder.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, verbose=1)
    
    # Evaluate the encoder on the test data
    test_loss = new_encoder.evaluate(X_test, X_test, verbose=1)
    
    print(f'Test MSE: {test_loss}')
    
    return new_encoder

def domain_adaptation_simple(file_path):
    latent_dim = 64
    outcome_df = pd.read_excel(file_path, sheet_name='outcomes')
    scanner_ids = pd.read_excel(file_path, sheet_name='scannerid')
    fmrifeatures_df = pd.read_excel(file_path, sheet_name='rawfmrifeatures')


    # Subset scanner IDs for EP1 and EP2
    subset_scanner_ids_EP1 = scanner_ids[scanner_ids['EP1or2'] == 1]
    subset_scanner_ids_EP2 = scanner_ids[scanner_ids['EP1or2'] == 2]


    # Get usable outcomes and features for EP1 and EP2
    usable_outcomes_EP1 = outcome_df[outcome_df['SID'].isin(subset_scanner_ids_EP1['SID'])]
    usable_outcomes_EP2 = outcome_df[outcome_df['SID'].isin(subset_scanner_ids_EP2['SID'])]
    
    fmrifeatures_df_EP1 = fmrifeatures_df[fmrifeatures_df['SID'].isin(subset_scanner_ids_EP1['SID'])]
    fmrifeatures_df_EP2 = fmrifeatures_df[fmrifeatures_df['SID'].isin(subset_scanner_ids_EP2['SID'])]

    # Calculate cue differences (assuming you have this function)
    cueB_minus_cueA_EP1 = calculate_cue_differences(fmrifeatures_df_EP1)
    cueB_minus_cueA_EP2 = calculate_cue_differences(fmrifeatures_df_EP2)

    ep1_data = merge_data(usable_outcomes_EP1, cueB_minus_cueA_EP1)
    ep2_data = merge_data(usable_outcomes_EP2, cueB_minus_cueA_EP2)

    print(ep1_data)
    print(ep2_data)

    scaler = MinMaxScaler()

    # Apply MinMax scaling to the data
    ep1_data_normalized = scaler.fit_transform(ep1_data)
    ep2_data_normalized = scaler.transform(ep2_data)  # Use transform on EP2 to avoid data leakage

    ep1_data = pd.DataFrame(ep1_data_normalized, columns=ep1_data.columns)
    ep2_data = pd.DataFrame(ep2_data_normalized, columns=ep2_data.columns)


    # Step 1: Train an encoder-decoder on EP1 data to reconstruct EP1 data
    encoder = build_encoder(ep1_data, latent_dim)
    decoder = build_decoder(ep1_data, latent_dim)
    # Train the encoder-decoder on EP1 data to reconstruct EP1 features (autoencoder-style)
    print("testing ep1 data with raw encoder decoder")
    model, encoder, decoder = train_encoder_decoder(encoder, decoder, ep1_data, ep1_data, epochs=10)  # Reconstruct EP1 features

    # Save the trained decoder

    encoder2 = build_encoder(ep2_data, latent_dim)
    decoder2 = build_decoder(ep2_data, latent_dim)
    # Train the encoder-decoder on EP1 data to reconstruct EP2 features (autoencoder-style)
    print("testing ep2 data with a raw encoder decoder")
    model2, encoder2, decoder2 = train_encoder_decoder(encoder2, decoder2, ep2_data, ep2_data, epochs=10)  # Reconstruct EP2 features

    save_decoder(decoder, 'trained_decoder.keras')

    # Step 2: Create a new encoder for EP2, using the saved decoder and freezing its weights
    new_encoder = create_new_encoder(ep2_data, decoder, latent_dim)

    # Fine-tune the new model (encoder for EP2) on EP2 data
    new_encoder.compile(optimizer='adam', loss='mse')


    fine_tune_encoder(new_encoder,ep2_data)
    # The new_encoder should now be trained for domain adaptation tasks from EP2 data
    return new_encoder

def main(file_path):
    """Main function to run the analysis 10 times and return average results."""
    accuracies_no_demo = []
    aucs_no_demo = []
    conf_matrices_no_demo = []

    accuracies_with_demo = []
    aucs_with_demo = []
    conf_matrices_with_demo = []

    accuracies_no_demo_ep2 = []
    aucs_no_demo_ep2 = []
    conf_matrices_no_demo_ep2 = []

    accuracies_with_demo_ep2 = []
    aucs_with_demo_ep2 = []
    conf_matrices_with_demo_ep2 = []

    # Run 10 times
    for _ in range(10):
        usable_outcomes, fmrifeatures_df = load_data(file_path)
        cueB_minus_cueA = calculate_cue_differences(fmrifeatures_df)
        merged_df = merge_data(usable_outcomes, cueB_minus_cueA)

        # Model without demographics
        X, y = prepare_data(merged_df)
        average_accuracy, auc, conf_matrix = evaluate_model(X, y)
        
        # Store the results
        accuracies_no_demo.append(average_accuracy)
        aucs_no_demo.append(auc)
        conf_matrices_no_demo.append(conf_matrix)

        # Model with demographics
        merged_df_with_demos = merge_demographics_data(usable_outcomes, cueB_minus_cueA, file_path)
        X_demos, y_demos = prepare_data(merged_df_with_demos)
        average_accuracy_demos, auc_demos, conf_matrix_demos = evaluate_model(X_demos, y_demos)

        accuracies_with_demo.append(average_accuracy_demos)
        aucs_with_demo.append(auc_demos)
        conf_matrices_with_demo.append(conf_matrix_demos)


        # EP2 - Same steps for second part of the analysis
        usable_outcomes, fmrifeatures_df = load_data2(file_path)
        cueB_minus_cueA = calculate_cue_differences(fmrifeatures_df)
        merged_df = merge_data(usable_outcomes, cueB_minus_cueA)

        # Model without demographics
        X, y = prepare_data(merged_df)
        average_accuracy, auc, conf_matrix = evaluate_model(X, y)

        # Store the results for EP2
        accuracies_no_demo_ep2.append(average_accuracy)
        aucs_no_demo_ep2.append(auc)
        conf_matrices_no_demo_ep2.append(conf_matrix)

        # Model with demographics
        merged_df_with_demos = merge_demographics_data(usable_outcomes, cueB_minus_cueA, file_path)
        X_demos, y_demos = prepare_data(merged_df_with_demos)
        average_accuracy_demos, auc_demos, conf_matrix_demos = evaluate_model(X_demos, y_demos)

        accuracies_with_demo_ep2.append(average_accuracy_demos)
        aucs_with_demo_ep2.append(auc_demos)
        conf_matrices_with_demo_ep2.append(conf_matrix_demos)

    # Average results
    avg_accuracy_no_demo = np.mean(accuracies_no_demo)
    avg_auc_no_demo = np.mean(aucs_no_demo)
    avg_conf_matrix_no_demo = np.mean(conf_matrices_no_demo, axis=0)

    avg_accuracy_with_demo = np.mean(accuracies_with_demo)
    avg_auc_with_demo = np.mean(aucs_with_demo)
    avg_conf_matrix_with_demo = np.mean(conf_matrices_with_demo, axis=0)

    avg_accuracy_no_demo_ep2 = np.mean(accuracies_no_demo_ep2)
    avg_auc_no_demo_ep2 = np.mean(aucs_no_demo_ep2)
    avg_conf_matrix_no_demo_ep2 = np.mean(conf_matrices_no_demo_ep2, axis=0)

    avg_accuracy_with_demo_ep2 = np.mean(accuracies_with_demo_ep2)
    avg_auc_with_demo_ep2 = np.mean(aucs_with_demo_ep2)
    avg_conf_matrix_with_demo_ep2 = np.mean(conf_matrices_with_demo_ep2, axis=0)

    # Print average results
    print(f"Average LOO Accuracy (without demographics): {avg_accuracy_no_demo:.4f}")
    print(f"AUC (without demographics): {avg_auc_no_demo:.4f}")
    print("Confusion Matrix (without demographics):")
    print(avg_conf_matrix_no_demo)

    print(f"\nAverage LOO Accuracy (with demographics): {avg_accuracy_with_demo:.4f}")
    print(f"AUC (with demographics): {avg_auc_with_demo:.4f}")
    print("Confusion Matrix (with demographics):")
    print(avg_conf_matrix_with_demo)

    print(f"\nEP2 Average LOO Accuracy (without demographics): {avg_accuracy_no_demo_ep2:.4f}")
    print(f"EP2 AUC (without demographics): {avg_auc_no_demo_ep2:.4f}")
    print("EP2 Confusion Matrix (without demographics):")
    print(avg_conf_matrix_no_demo_ep2)

    print(f"\nEP2 Average LOO Accuracy (with demographics): {avg_accuracy_with_demo_ep2:.4f}")
    print(f"EP2 AUC (with demographics): {avg_auc_with_demo_ep2:.4f}")
    print("EP2 Confusion Matrix (with demographics):")
    print(avg_conf_matrix_with_demo_ep2)

def main_with_pca(file_path, n_components=33):
    """Main function to run the analysis using PCA for dimensionality reduction."""
    usable_outcomes, fmrifeatures_df = load_data(file_path)
    cueB_minus_cueA = calculate_cue_differences(fmrifeatures_df)
    merged_df = merge_data(usable_outcomes, cueB_minus_cueA)

    # Model without demographics, with PCA
    X_pca, y = prepare_data_with_pca(merged_df, n_components)
    average_accuracy_pca, auc_pca, conf_matrix_pca = evaluate_model(X_pca, y)
    
    # Model with demographics, with PCA
    merged_df_with_demos = merge_demographics_data(usable_outcomes, cueB_minus_cueA, file_path)
    X_demos_pca, y_demos = prepare_data_with_pca(merged_df_with_demos, n_components)
    average_accuracy_demos_pca, auc_demos_pca, conf_matrix_demos_pca = evaluate_model(X_demos_pca, y_demos)

    # Print results
    print(f"Average LOO Accuracy (without demographics, PCA): {average_accuracy_pca:.4f}")
    print(f"AUC (without demographics, PCA): {auc_pca:.4f}")
    print("Confusion Matrix (without demographics, PCA):")
    print(conf_matrix_pca)

    print(f"\nAverage LOO Accuracy (with demographics, PCA): {average_accuracy_demos_pca:.4f}")
    print(f"AUC (with demographics, PCA): {auc_demos_pca:.4f}")
    print("Confusion Matrix (with demographics, PCA):")
    print(conf_matrix_demos_pca)



#Run the main function with the specified file path
file_path = '/Users/alexd/Documents/Davidson Research/WholeBrainBetas.xlsx'

domain_adaptation_simple(file_path)
main(file_path)
#main_with_pca(file_path, n_components=33)



#results over 10 runs

# Average LOO Accuracy (without demographics): 0.5981
# AUC (without demographics): 0.6618
# Confusion Matrix (without demographics):
# [[14.5  8.5]
#  [12.4 16.6]]

# Average LOO Accuracy (with demographics): 0.6509
# AUC (with demographics): 0.7333
# Confusion Matrix (with demographics):
# [[16.2  7.8]
#  [12.1 20.9]]

# EP2 Average LOO Accuracy (without demographics): 0.4600
# EP2 AUC (without demographics): 0.4782
# EP2 Confusion Matrix (without demographics):
# [[5.4 6.6]
#  [9.6 8.4]]

# EP2 Average LOO Accuracy (with demographics): 0.6543
# EP2 AUC (with demographics): 0.7263
# EP2 Confusion Matrix (with demographics):
# [[ 9.8  5.2]
#  [ 6.9 13.1]]




# testing ep1 data with raw encoder decoder
# Epoch 1/10
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - loss: 0.0496  
# Epoch 2/10
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 537us/step - loss: 0.0475
# Epoch 3/10
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 475us/step - loss: 0.0470
# Epoch 4/10
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 499us/step - loss: 0.0455
# Epoch 5/10
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 528us/step - loss: 0.0438
# Epoch 6/10
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 525us/step - loss: 0.0433
# Epoch 7/10
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 556us/step - loss: 0.0428
# Epoch 8/10
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 495us/step - loss: 0.0422
# Epoch 9/10
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 499us/step - loss: 0.0397
# Epoch 10/10
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 479us/step - loss: 0.0384
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 32ms/step - loss: 0.0420
# Test MSE: 0.0420086644589901 #######RESULTS
# /Users/alexd/aiResearch/lib/python3.9/site-packages/keras/src/layers/core/input_layer.py:26: UserWarning: Argument `input_shape` is deprecated. Use `shape` instead.
#   warnings.warn(
# testing ep2 data with a raw encoder decoder
# Epoch 1/10
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 391ms/step - loss: 0.0511
# Epoch 2/10
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 0.0501
# Epoch 3/10
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 0.0496
# Epoch 4/10
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 0.0493
# Epoch 5/10
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 0.0489
# Epoch 6/10
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 0.0485
# Epoch 7/10
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 0.0480
# Epoch 8/10
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 0.0474
# Epoch 9/10
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 0.0466
# Epoch 10/10
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 0.0458
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 32ms/step - loss: 0.0456
# Test MSE: 0.04558004066348076
# /Users/alexd/aiResearch/lib/python3.9/site-packages/keras/src/layers/core/input_layer.py:26: UserWarning: Argument `input_shape` is deprecated. Use `shape` instead.
#   warnings.warn(
# Epoch 1/10
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 245ms/step - loss: 0.0481
# Epoch 2/10
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 0.0474
# Epoch 3/10
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 0.0466
# Epoch 4/10
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 0.0459
# Epoch 5/10
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 0.0452
# Epoch 6/10
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 0.0447
# Epoch 7/10
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 0.0443
# Epoch 8/10
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 0.0439
# Epoch 9/10
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 0.0436
# Epoch 10/10
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 0.0435
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 32ms/step - loss: 0.0435
# Test MSE: 0.04351582005620003


#[52 rows x 174 columns] EP1
#[30 rows x 174 columns] EP2
