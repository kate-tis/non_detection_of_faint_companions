import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Orbit:
    def __init__(self, radius, phi_0=0.0, inclination=0.0, position_angle=0.0, star_mass=2.0):
        # Constants
        self.G = 39.478  # Gravitational constant in AU^3 / (solar mass * year^2)
        self.solar_mass = 1.0  # Solar mass in solar mass units

        # Star parameters
        self.star_mass = star_mass  # Mass of the star in solar masses

        # Orbit parameters
        self.radius = radius  # Radius of orbit in AU
        self.phi_0 = np.radians(phi_0)  # Initial angular position in radians
        self.inclination = np.radians(inclination)  # Inclination angle in radians
        self.position_angle = np.radians(position_angle)  # Position angle in radians

        # Orbital period in years
        self.orbital_period = 2 * np.pi * np.sqrt(self.radius**3 / (self.G * self.star_mass))

        # Orbital velocity in AU/year
        self.velocity = np.sqrt(self.G * self.star_mass / self.radius)

    def __str__(self):
        basic_info = (
            f"Orbit Information:\n"
            f"  Radius: {self.radius} AU\n"
            f"  Initial Angle (phi_0): {np.degrees(self.phi_0):.2f} degrees\n"
            f"  Inclination: {np.degrees(self.inclination):.2f} degrees\n"
            f"  Position Angle: {np.degrees(self.position_angle):.2f} degrees\n"
        )
        additional_info = (
            f"Star and Orbit Details:\n"
            f"  Star Mass: {self.star_mass} Solar Masses\n"
            f"  Orbital Period: {self.orbital_period:.2f} years\n"
            f"  Orbital Velocity: {self.velocity:.2f} AU/year"
        )
        return f"{basic_info}\n{additional_info}"

    def position_at_time(self, time):
        """
        Calculate the position (x, y, z) at a given time.
        :param time: Time in years
        :return: Array of shape (3,) for (x, y, z) in AU
        """
        delta_phi = self.velocity * time / self.radius
        return self.position_at_delta_phi(delta_phi)

    def position_at_delta_phi(self, delta_phi):
        """
        Calculate the position (x, y, z) after moving by delta_phi from the initial position.
        :param delta_phi: Change in the orbital position angle in radians
        :return: Array of shape (n, 3) or (3,) for (x, y, z) in AU
        """
        phi = self.phi_0 + delta_phi

        # Calculate the unrotated position
        x = self.radius * np.cos(phi)
        y = self.radius * np.sin(phi)
        z = np.zeros_like(phi)  # Ensure z has the same shape as phi

        # Rotate for inclination angle (rotation about x-axis)
        y_rot = y * np.cos(self.inclination) - z * np.sin(self.inclination)
        z_rot = y * np.sin(self.inclination) + z * np.cos(self.inclination)
        y = y_rot
        z = z_rot

        # Rotate for position angle (rotation about z-axis, defining orientation in the sky)
        x_rot = x * np.cos(self.position_angle) - y * np.sin(self.position_angle)
        y_rot = x * np.sin(self.position_angle) + y * np.cos(self.position_angle)
        x = x_rot
        y = y_rot

        # Return array of shape (n, 3) or (3,)
        if np.ndim(phi) > 0:
            return np.column_stack((x, y, z))
        else:
            return np.array([x, y, z])

    def sky_projection(self, time):
        """
        Calculate the projection of the position on the x-y-plane (sky projection).
        :param time: Time in years
        :return: Array of shape (2,) for (x_proj, y_proj) in AU
        """
        delta_phi = self.velocity * time / self.radius
        return self.sky_projection_at_delta_phi(delta_phi)

    def sky_projection_at_delta_phi(self, delta_phi):
        """
        Calculate the projection of the position on the x-y-plane (sky projection) after moving by delta_phi.
        :param delta_phi: Change in the orbital position angle in radians
        :return: Array of shape (n, 2) or (2,) for (x_proj, y_proj) in AU
        """
        positions = self.position_at_delta_phi(delta_phi)
        if np.ndim(delta_phi) > 0:
            x = positions[:, 0]
            y = positions[:, 1]
            return np.column_stack((x, y))
        else:
            return positions[:2]

    def visualize_orbit(self, phi_start=None, phi_end=None, time_points=100, arrow_scale=1.0):
        """
        Visualize the orbit and the sky projection.
        :param phi_start: Starting angle of the phi range in radians
        :param phi_end: Ending angle of the phi range in radians
        :param time_points: Number of time points for visualization
        :param arrow_scale: Scaling factor for the direction arrow length
        """
        # Set default values for phi_start and phi_end if not provided
        if phi_start is None:
            phi_start = 0.0
        if phi_end is None:
            phi_end = 2 * np.pi

        delta_phis = np.linspace(phi_start, phi_end, time_points)
        positions = self.position_at_delta_phi(delta_phis)

        # Extract x, y, z coordinates
        x_vals, y_vals, z_vals = positions[:, 0], positions[:, 1], positions[:, 2]

        # Create 3D plot of the orbit
        fig = plt.figure(figsize=(12, 6))
        ax_3d = fig.add_subplot(121, projection='3d')
        ax_3d.plot(x_vals, y_vals, z_vals, label="Planet Orbit")
        ax_3d.set_xlabel('X [AU]')
        ax_3d.set_ylabel('Y [AU]')
        ax_3d.set_zlabel('Z [AU]')
        ax_3d.set_title('3D Orbit of the Planet')
        ax_3d.legend()

        # Create 2D projection plot (sky projection)
        sky_projections = self.sky_projection_at_delta_phi(delta_phis)
        x_proj_vals, y_proj_vals = sky_projections[:, 0], sky_projections[:, 1]
                # Add an arrow to indicate the direction of motion
        arrow_dx = (x_proj_vals[1] - x_proj_vals[0]) * arrow_scale
        arrow_dy = (y_proj_vals[1] - y_proj_vals[0]) * arrow_scale
        ax_2d.arrow(x_proj_vals[0], y_proj_vals[0], arrow_dx, arrow_dy, 
                    head_width=0.05 * arrow_scale, head_length=0.05 * arrow_scale, 
                    fc='black', ec='black', label='Direction')
        ax_2d = fig.add_subplot(122)
        ax_2d.plot(x_proj_vals, y_proj_vals, label="Sky Projection")
        ax_2d.set_xlabel('X [AU]')
        ax_2d.set_ylabel('Y [AU]')
        ax_2d.set_title('Sky Projection (x-y plane)')
        ax_2d.axis('equal')

        # Mark the initial position with a black circle
        ax_2d.plot(x_proj_vals[0], y_proj_vals[0], 'ko', label='Initial Position')



        ax_2d.legend()
        plt.tight_layout()
        plt.show()

def analyze_orbit_on_map(orbit, num_positions, pixel_width, sample_map, label = r'$\Delta V_{\rm max}$',arrow_scale=1.0, plot = True):
    """
    Analyze the orbital positions on a pixel map.
    :param orbit: Orbit instance
    :param num_positions: Number of positions to evaluate along the orbit
    :param pixel_width: Width of each pixel in AU
    :param sample_map: 2D array of shape (N, N) representing the flux map
    :param arrow_scale: Scaling factor for the direction arrow length
    """
    N = sample_map.shape[0]
    assert sample_map.shape == (N, N), "Sample map must be a square 2D array."

    delta_phis = np.linspace(0, 2 * np.pi, num_positions)
    map_values = []

    for orbital_position in delta_phis:
        # Calculate position x, y in AU
        xy_position = orbit.sky_projection_at_delta_phi(orbital_position)

        # Determine corresponding pixel index in map
        pixel_index_x = int(np.floor(xy_position[0] / pixel_width)) + N // 2
        pixel_index_y = int(np.floor(xy_position[1] / pixel_width)) + N // 2

        # Check if the pixel index is within bounds
        if 0 <= pixel_index_x < N and 0 <= pixel_index_y < N:
            # Read the pixel value at xy_position in map (from sample_map)
            pixel_value = sample_map[pixel_index_x, pixel_index_y]
        else:
            # If out of bounds, set the pixel value to zero
            pixel_value = 0

        # Store result in map_values list
        map_values.append(pixel_value)

    if plot == True:
        # Plot map_values as a function of the orbital position
        plt.figure()
        plt.plot(delta_phis, map_values, '-o')
        plt.xlabel('Orbital Position (radians)')
        plt.ylabel(label)
        plt.title('Function of Orbital Position')
        plt.grid(True)
        plt.show()
    
        # Plot flux map and overlay orbit positions
        plt.figure()
        plt.imshow(sample_map.T, origin='lower', 
                   extent=[-N//2 * pixel_width, N//2 * pixel_width, -N//2 * pixel_width, N//2 * pixel_width], 
                   cmap='viridis')
        plt.colorbar(label=label)
        x_vals, y_vals = zip(*[orbit.sky_projection_at_delta_phi(orbital_position) for orbital_position in delta_phis])
        plt.plot(x_vals, y_vals, 'r-', label='Orbit Path')
        plt.plot(x_vals[0], y_vals[0], 'ko', label='Initial Position')
        arrow_dx = (x_vals[1] - x_vals[0]) * arrow_scale
        arrow_dy = (y_vals[1] - y_vals[0]) * arrow_scale
        plt.arrow(x_vals[0], y_vals[0], arrow_dx, arrow_dy, head_width=0.01, head_length=0.01, fc='black', ec='black', label='Direction')
        plt.xlabel('X [AU]')
        plt.ylabel('Y [AU]')
        plt.title('Map with Orbit Path Overlay')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    return(map_values)
