/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <limits>

#include "helper_functions.h"

using std::string;
using std::vector;
using std::normal_distribution;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 100;  // TODO: Set the number of particles
  std::default_random_engine gen;

  // create normal distributions for x, y, and theta
  normal_distribution<double> dist_x(x, std[0]);  
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  // create particles in loop with total number or particles equal to num_particles
  for (int i = 0; i < num_particles; ++i) {
    double particle_x, particle_y, particle_theta;

    // create particle x, y, and theta from a random normal distribution
    particle_x = dist_x(gen);
    particle_y = dist_y(gen);
    particle_theta = dist_theta(gen);

    // create a particle instance and add it to the particles vector
    Particle p{i, particle_x, particle_y, particle_theta, 1.0};
    particles.push_back(p);
    weights.push_back(1.0);
  }
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  std::default_random_engine gen;

  // create normal distributions for x, y, and theta
  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_theta(0, std_pos[2]);

  for (int i = 0; i < particles.size(); i++) {
    double theta = particles[i].theta;

    if (fabs(yaw_rate) > 0.0001) {
      particles[i].x += velocity / yaw_rate * (sin(theta + yaw_rate * delta_t) - sin(theta));
      particles[i].y += velocity / yaw_rate * (cos(theta) - cos(theta + yaw_rate * delta_t));
      particles[i].theta += yaw_rate * delta_t;
    } else {
      particles[i].x += velocity * delta_t * cos(theta);
      particles[i].y += velocity * delta_t * sin(theta);
    }

    particles[i].x += dist_x(gen);
    particles[i].y += dist_y(gen);
    particles[i].theta += dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  // for (int o = 0; o < observations.size(); o++) {
  //   // LandmarkObs closestObs = observations[0];
  //   float closestDistance = dist(predicted[0].x, predicted[0].y, observations[o].x, observations[o].y);
  //   float distance = 0.0;

  //   for (int p = 1; p < predicted.size(); p++) {
  //     distance = dist(predicted[p].x, predicted[p].y, observations[o].x, observations[o].y);

  //     if (distance < closestDistance) {
  //       // closestObs = observations[o];
  //       observations[o].id = predicted[p].id;
  //       closestDistance = distance;
  //     }
  //   }
  // }
	for (int i = 0; i < observations.size(); ++i) {
	    
	    // Current observation
	    LandmarkObs obs_curr = observations[i];

	    // Initialize minimum distance to max value
	    double dist_min = std::numeric_limits<double>::max();

	    // Initialize map_id
	    int map_id = -1;
	    
	    for (unsigned int j = 0; j < predicted.size(); ++j) {
	      // Current prediction
	      LandmarkObs pred_curr = predicted[j];
	      
	      // Calculate distance between current ith observation and jth landmark
	      double dist_curr = dist(obs_curr.x, obs_curr.y, pred_curr.x, pred_curr.y);

	      // Store id of the nearest landmark
	      if (dist_curr < dist_min) {
	        dist_min = dist_curr;
	        map_id = pred_curr.id;
	      }
	    }
	    // Link observation id to that of nearest landmark
	    observations[i].id = map_id;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  for (int p = 0; p < particles.size(); p++) {
    double particle_x = particles[p].x;
    double particle_y = particles[p].y;
    double particle_theta = particles[p].theta;
    double particle_w = 1;
    particles[p].weight = particle_w;
    
    // vector of transformed observations into map coordinates
    vector<LandmarkObs> observations_map_coords;

    // transform each observation into map coordinates
    for (int o = 0; o < observations.size(); o++) {
      LandmarkObs mapped_obs;
      mapped_obs.id = observations[o].id;
      mapped_obs.x = observations[o].x*cos(particle_theta) - observations[o].y*sin(particle_theta) + particle_x;
      mapped_obs.y = observations[o].x*sin(particle_theta) + observations[o].y*cos(particle_theta) + particle_y;
      observations_map_coords.push_back(mapped_obs);
    }

    // vector of landmarks in range of sensor
    vector<LandmarkObs> predictions;

    // select only landmarks within range of sensor
    for (int l = 0; l < map_landmarks.landmark_list.size(); l++) {
      LandmarkObs landmark;
      landmark.id = map_landmarks.landmark_list[l].id_i;
      landmark.x = map_landmarks.landmark_list[l].x_f;
      landmark.y = map_landmarks.landmark_list[l].y_f;

      if (dist(landmark.x, landmark.y, particle_x, particle_y) <= sensor_range) {
        predictions.push_back(landmark);
      }
    }

    // find predicted measurement that's closest to the observed measurement
    dataAssociation(predictions, observations_map_coords);

    // update particle weights based on particle observations vs actual observations
    for (int o = 0; o < observations_map_coords.size(); o++) {

      double obs_x = observations_map_coords[o].x;
      double obs_y = observations_map_coords[o].y;
      double sig_x = std_landmark[0];
      double sig_y = std_landmark[1];
      double landmark_x, landmark_y;

      for (int i = 0; i < predictions.size(); i++) {
        if (observations_map_coords[o].id == predictions[i].id) {
          landmark_x = predictions[i].x;
          landmark_y = predictions[i].y;
        }
      }

      // calculate weight of each particle using the Multivariate_Gaussian probability density function
      particle_w = ( 1/(2*M_PI*sig_x*sig_y)) * exp( -( pow(landmark_x-obs_x,2)/(2*pow(sig_x, 2)) + (pow(landmark_y-obs_y,2)/(2*pow(sig_y, 2))) ) );
      // update current particle weight
      particles[p].weight *= particle_w;
    }

    // update weights vector
    weights[p] = particles[p].weight;
  }
  std::cout << "actual num particles  " << particles.size() << std::endl;
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  std::default_random_engine gen;
  std::discrete_distribution<int> dist(weights.begin(), weights.end());

  vector<Particle> particles_resampled;

  for (int p = 0; p < num_particles; p++) {
    particles_resampled.push_back(particles[dist(gen)]);
  }
  std::cout << "resampled HERE (" << particles.size() << ")" << std::endl;

  // replace particles with resampled particles
  particles = particles_resampled;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}