# filename: iris_model.py
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Step 1: Import the necessary libraries

# Step 2: Load and preprocess the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert the numpy arrays to PyTorch tensors
X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).long()
X_test = torch.from_numpy(X_test).float()

# Step 3: Define the neural network architecture
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(4, 16)  # 4 input features, 16 hidden units in the first layer
        self.fc2 = nn.Linear(16, 8)  # 16 hidden units in the first layer, 8 hidden units in the second layer
        self.fc3 = nn.Linear(8, 3)   # 8 hidden units in the second layer, 3 output classes
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = MyModel()

# Step 4: Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Step 5: Train the model
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Step 6: Predict the species of the given data point
model.eval()
data_point = torch.tensor([[5.1, 3.5, 1.4, 0.2]])  # Convert the data point to a PyTorch tensor
predicted_species = torch.argmax(model(data_point), dim=1).item()

species_mapping = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
predicted_species_name = species_mapping[predicted_species]
print(f"Predicted species: {predicted_species_name}")
