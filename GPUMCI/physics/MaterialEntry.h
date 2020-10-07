#pragma once
#include <stdexcept>
#include <vector>
#include <Eigen/Core>
#include <odl_cpp_utils/utils/cast.h>

namespace gpumci {
/*
	 Holder for material attenuation coefficient data

	 Data begins at e=0 and increases by energyStep for each entry.

	 Each row represents one material.

	 Energies are given in MeV	
	*/
struct MaterialData {
    MaterialData(double energyStep_,
                 const Eigen::ArrayXXd& compton_,
                 const Eigen::ArrayXXd& rayleigh_,
                 const Eigen::ArrayXXd& photo_)
        : energyStep(energyStep_),
          n_energies(narrow_cast<unsigned>(compton_.cols())),
          n_materials(narrow_cast<unsigned>(compton_.rows())),
          compton(compton_),
          rayleigh(rayleigh_),
          photo(photo_) {
        if (rayleigh.rows() != compton.rows() ||
            rayleigh.cols() != compton.cols() ||
            photo.rows() != compton.rows() ||
            photo.cols() != compton.cols()) {
            throw std::invalid_argument("Sizes of compton, rayleigh and photo tables have to match");
        }
    }

    const double energyStep;
    const unsigned n_energies;
    const unsigned n_materials;
    const Eigen::ArrayXXd compton;
    const Eigen::ArrayXXd rayleigh;
    const Eigen::ArrayXXd photo;
};
}
