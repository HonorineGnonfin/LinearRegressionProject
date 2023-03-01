# Instanciate the LinearRegression class 
model_nor=LinearRegression("normalEquation")
model_qr=LinearRegression("qrFactorization")

# Train the model
model_nor.fit(X_train,y_train)
model_qr.fit(X_train,y_train)

# print the learned theta
theta_nor=model_nor.theta
theta_qr=model_qr.theta
print(f" theta paramatre for normalEquation {theta_nor}")
print(f" theta paramatre for qrFactorizatio {theta_qr}")

