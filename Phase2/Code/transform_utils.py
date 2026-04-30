import torch
import torch.nn.functional as F

    # Skew-symmetric matrices for omega
def skew(w):
    z = torch.zeros_like(w[:, 0])
    return torch.stack([
        torch.stack([z, -w[:, 2], w[:, 1]], dim=1),
        torch.stack([w[:, 2], z, -w[:, 0]], dim=1),
        torch.stack([-w[:, 1], w[:, 0], z], dim=1)
    ], dim=1)


def batch_se3_exp(xi):
    """
    Converts a batch of se(3) vectors to SE(3) matrices.
    xi: (B, 6) -> [translation_v, rotation_omega]
    """
    B = xi.shape[0]
    v = xi[:, :3]
    omega = xi[:, 3:]
    theta = torch.norm(omega, dim=1, keepdim=True)

    Omega = skew(omega)
    I = torch.eye(3).to(xi.device).unsqueeze(0).expand(B, 3, 3)
    
    # Rodrigues for Rotation
    # Use Taylor expansion for small theta to avoid NaN
    mask = (theta > 1e-4).float().view(B, 1, 1)
    
    # Standard Rodrigues
    Omega2 = torch.bmm(Omega, Omega)
    R = I + (torch.sin(theta)/theta).view(B, 1, 1) * Omega + \
        ((1 - torch.cos(theta))/(theta**2)).view(B, 1, 1) * Omega2
        
    # Standard V matrix for Translation
    V = I + ((1 - torch.cos(theta))/(theta**2)).view(B, 1, 1) * Omega + \
        ((theta - torch.sin(theta))/(theta**3)).view(B, 1, 1) * Omega2

    # Handle very small rotations (Identity)
    R = mask * R + (1 - mask) * (I + Omega)
    V = mask * V + (1 - mask) * (I + 0.5 * Omega)

    # Construct SE(3) (4x4)
    T = torch.zeros((B, 4, 4), device=xi.device)
    T[:, :3, :3] = R
    T[:, :3, 3] = torch.bmm(V, v.unsqueeze(2)).squeeze(2)
    T[:, 3, 3] = 1.0
    return T

def batch_dpose(delta_pose, prev_pose):
    quats = prev_pose[:,0,3:]
    rots = quaternion_to_matrix(quats)
    bottom_row = torch.tensor([0,0,0,1]).expand(delta_pose.shape[0], 1, 4).to(delta_pose.device)
    prev_ts = prev_pose[:,[0],:3].permute(0,2,1)
    prev_Ts = torch.cat((torch.cat((rots, prev_ts), dim=2), bottom_row), dim=1)
    new_Ts = torch.bmm(prev_Ts, delta_pose)
    new_quats = matrix_to_quaternion(new_Ts[:,:3,:3])
    new_poses = torch.cat((new_Ts[:,:3,3], new_quats), dim=1).unsqueeze(1)
    return new_poses
    

def quaternion_to_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret

def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(*batch_dim, 9), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    # pyre-ignore [16]: `torch.Tensor` has no attribute `new_tensor`.
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(q_abs.new_tensor(0.1)))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :  # pyre-ignore[16]
    ].reshape(*batch_dim, 4)