# Make a prediction on X_test
model_nor.predict(X_test)
y_pred_nor=model_nor.y_pred
#print(y_test)
#print(y_pred_nor)
model_qr.predict(X_test)
y_pred_qr=model_qr.y_pred
#print(y_test)
#print(y_pred_qr)

# Compute the MSE (Evaluate both, regression and classification)
mse_nor=mse(y_test, y_pred_nor)
mse_qr=mse(y_test, y_pred_qr)

#MSE for normal Equation
print(f"MSE for normal Equation is: {mse_nor}")

#MSE for normal Qrfactorization
print(f"MSE for Qrfactorization  is: {mse_qr}")

