import math
import numpy as np
import matplotlib.pyplot as plt
#probably need to compile all this into a function at some point so you can call it. 


def create_rotation_matrix(theta, axis:np.ndarray):
    #make n direction vector a normalized 1D unit vector. 
    n = np.array(axis, dtype=float) / np.linalg.norm(axis)

    q0 = math.cos(theta/2.0)
    q1, q2, q3 = (n * math.sin(theta/2.0)).flatten()
    #q_vec containes q1, q2, q3
    #to create q which encompasses both unit vector 
    q = np.array([[q0], [q1], [q2], [q3]])
    C_q = np.array([[1-2*q2**2 - 2*q3**2, 2*q1*q2 - 2*q0*q3, 2*q1*q3 + 2*q0*q2], [2*q1*q2 + 2*q0*q3, 1-2*q1**2 - 2*q3**2, 2*q2*q3 - 2*q0*q1], [2*q1*q3 - 2*q0*q2, 2*q2*q3 + 2*q0*q1, 1-2*q1**2 - 2*q2**2]])

    qk0 = math.cos(theta/2.0)
    qk1, qk2, qk3 = (n * math.sin(theta/2.0)).flatten()
    qk = np.array([[qk0], [qk1], [qk2], [qk3]])
    v_q = np.array([q1,q2,q3])
    v_qk = np.array([qk1,qk2,qk3])
    q_transpose = v_q.T

    C_qk = np.array([[1-2*qk2**2 - 2*qk3**2, 2*qk1*qk2 - 2*qk0*qk3, 2*qk1*qk3 + 2*qk0*qk2], [2*qk1*qk2 + 2*qk0*qk3, 1-2*qk1**2 - 2*qk3**2, 2*qk2*qk3 - 2*qk0*qk1], [2*qk1*qk3 - 2*qk0*qk2, 2*qk2*qk3 + 2*qk0*qk1, 1-2*qk1**2 - 2*qk2**2]])

    C_k1 = C_q @ C_qk

#you have to divide by scalar part and vector part. 
    scalar = qk0*q0 - np.dot(q_transpose, v_qk)
    vector = q0*v_qk + qk0*v_q + np.cross(v_q, v_qk, axis=0)
    q_k1 = np.vstack([scalar, vector.reshape(3, 1)])

    return q, C_q, q_k1, C_k1

q, C_q, q_k1, C_k1 = create_rotation_matrix(math.pi/5, np.array([1/math.sqrt(3),1/math.sqrt(3),1/math.sqrt(3)]))
print("Quaternion q:\n", q)
print("\nRotation Matrix C_q:\n", C_q)
print("\nCompound Quaternion q_k1:\n", q_k1)
print("\nCompound Quaternion Matrix C_k1:\n", C_k1)

iterations = 1000
determinants = []
quaternion_norms = []

C_base = C_q.copy()  # Create a copy of the initial rotation matrix
C_k = C_q.copy()  # Initialize C_k with the initial rotation matrix

q_base = q.copy()  # Create a copy of the initial quaternion
q_k = q.copy()  # Initialize q_k with the initial quaternion


for n in range(iterations):
    # Update the rotation matrix by multiplying with the initial rotation matrix
    C_k = C_k @ C_base
    determinants.append(np.linalg.det(C_k))
    scalar_k, scalar_base = float(q_k[0, 0]), float(q_base[0, 0])
    vec_k = q_k[1:].flatten()
    vec_base = q_base[1:].flatten()
    
    # Calculate compound quaternion parts via Hamilton Product
    scalar_next = scalar_k * scalar_base - np.dot(vec_k, vec_base)
    vec_next = scalar_k * vec_base + scalar_base * vec_k + np.cross(vec_k, vec_base)
    
    # Re-stack into a (4, 1) column vector for the next loop iteration
    q_k = np.vstack([scalar_next, vec_next.reshape(3, 1)])
    q_norm = np.linalg.norm(q_k)
    quaternion_norms.append(q_norm)

x_axis = np.arange(1, len(determinants) + 1)
plt.figure(figsize=(12, 5))

# Plot Left: Matrix Determinant Drift
plt.subplot(1, 2, 1)
plt.plot(range(1, iterations + 1), determinants, color='blue', label=r'$\det(\mathbf{C}_k)$')
plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Ideal Det = 1.0')
plt.title('Rotation Matrix Determinant Drift')
plt.xlabel('Iteration (k)')
plt.ylabel('Determinant Value')
plt.ticklabel_format(useOffset=False, style='plain')  # Prevents confusing offset labels
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, iterations + 1), quaternion_norms, color='purple', label=r'$\|\mathbf{q}_k\|$')
plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Ideal Norm = 1.0')
plt.title('Quaternion Norm Drift')
plt.xlabel('Iteration (k)')
plt.ylabel('Norm Value')
plt.ticklabel_format(useOffset=False, style='plain')  # Prevents confusing offset labels
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()

plt.tight_layout()
plt.show()