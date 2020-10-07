#include <GPUMCI/implementations/AbsorbingMC.h>
#include <GPUMCI/implementations/SimpleMC.h>
#include <GPUMCI/implementations/PrecomputedMC.h>
#include <GPUMCI/implementations/ForwardProjector.h>
#include <GPUMCI/implementations/PhaseSpaceMC.h>
#include <GPUMCI/implementations/PhaseSpaceStorePhotonsMC.h>
#include <GPUMCI/implementations/DoseMC.h>
#include <GPUMCI/implementations/GainMC.h>
#include <GPUMCI/implementations/GainDoseMC.h>
#include <GPUMCI/implementations/SpectMC.h>

#include <GPUMCI/utils/CudaCosWeighting.h>

#include <boost/python.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/numeric.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

#include <iostream>
#include <vector>
#include <Eigen/Core>
#include <odl_cpp_utils/python/numpy_eigen.h>
#include <odl_cpp_utils/utils/EigenUtils.h>
#include <odl_cpp_utils/cuda/CudaMemory.h>

using namespace boost::python;
using namespace Eigen;

class AbsorbingMCPython {
  public:
    AbsorbingMCPython(Eigen::Vector3i volumeSize,
                      Eigen::Vector3d volumeOrigin,
                      Eigen::Vector3d voxelSize,
                      Eigen::Vector2i detectorSize)
        : _mc(volumeSize,
              volumeOrigin,
              voxelSize,
              detectorSize) {
    }

    void setData(uintptr_t densityDevice,
                 uintptr_t materialTypeDevice) {
        _mc.setData(reinterpret_cast<const float*>(densityDevice),
                    reinterpret_cast<const uint8_t*>(materialTypeDevice));
    }

    void project(Eigen::Vector3d sourcePosition,
                 Eigen::Vector3d detectorOrigin,
                 Eigen::Vector3d pixelDirectionU,
                 Eigen::Vector3d pixelDirectionV,
                 uintptr_t resultPtrDevice) const {
        _mc.project(sourcePosition,
                    detectorOrigin,
                    pixelDirectionU,
                    pixelDirectionV,
                    reinterpret_cast<float*>(resultPtrDevice));
    }

  private:
    gpumci::AbsorbingMC _mc;
};

class SimpleMCPython {
  public:
    SimpleMCPython(Eigen::Vector3i volumeSize,
                   Eigen::Vector3d volumeOrigin,
                   Eigen::Vector3d voxelSize,
                   Eigen::Vector2i detectorSize)
        : _mc(volumeSize,
              volumeOrigin,
              voxelSize,
              detectorSize) {
    }

    void setData(uintptr_t densityDevice,
                 uintptr_t materialTypeDevice) {
        _mc.setData(reinterpret_cast<const float*>(densityDevice),
                    reinterpret_cast<const uint8_t*>(materialTypeDevice));
    }

    void project(Eigen::Vector3d sourcePosition,
                 Eigen::Vector3d detectorOrigin,
                 Eigen::Vector3d pixelDirectionU,
                 Eigen::Vector3d pixelDirectionV,
                 uintptr_t resultPrimaryPtrDevice,
                 uintptr_t resultScatterPtrDevice) const {
        _mc.project(sourcePosition,
                    detectorOrigin,
                    pixelDirectionU,
                    pixelDirectionV,
                    reinterpret_cast<float*>(resultPrimaryPtrDevice),
                    reinterpret_cast<float*>(resultScatterPtrDevice));
    }

  private:
    gpumci::SimpleMC _mc;
};

class PrecomputedMCPython {
  public:
    PrecomputedMCPython(Eigen::Vector3i volumeSize,
                        Eigen::Vector3d volumeOrigin,
                        Eigen::Vector3d voxelSize,
                        Eigen::Vector2i detectorSize,
                        int n_runs,
                        gpumci::MaterialData data,
                        gpumci::InteractionTables rayleighTables,
                        gpumci::InteractionTables comptonTables)
        : _mc(volumeSize,
              volumeOrigin,
              voxelSize,
              detectorSize,
              n_runs,
              data,
              rayleighTables,
              comptonTables) {
    }

    void setData(uintptr_t densityDevice,
                 uintptr_t materialTypeDevice) {
        _mc.setData(reinterpret_cast<const float*>(densityDevice),
                    reinterpret_cast<const uint8_t*>(materialTypeDevice));
    }

    void project(Eigen::Vector3d sourcePosition,
                 Eigen::Vector3d detectorOrigin,
                 Eigen::Vector3d pixelDirectionU,
                 Eigen::Vector3d pixelDirectionV,
                 uintptr_t resultPrimaryPtrDevice,
                 uintptr_t resultScatterPtrDevice) const {
        _mc.project(sourcePosition,
                    detectorOrigin,
                    pixelDirectionU,
                    pixelDirectionV,
                    reinterpret_cast<float*>(resultPrimaryPtrDevice),
                    reinterpret_cast<float*>(resultScatterPtrDevice));
    }

  private:
    gpumci::PrecomputedMC _mc;
};

class ForwardProjectorPython {
  public:
    ForwardProjectorPython(Eigen::Vector3i volumeSize,
                           Eigen::Vector3d volumeOrigin,
                           Eigen::Vector3d voxelSize,
                           Eigen::Vector2i detectorSize,
                           gpumci::MaterialData data)
        : _mc(volumeSize,
              volumeOrigin,
              voxelSize,
              detectorSize,
              data) {
    }

    void setData(uintptr_t densityDevice,
                 uintptr_t materialTypeDevice) {
        _mc.setData(reinterpret_cast<const float*>(densityDevice),
                    reinterpret_cast<const uint8_t*>(materialTypeDevice));
    }

    void project(Eigen::Vector3d sourcePosition,
                 Eigen::Vector3d detectorOrigin,
                 Eigen::Vector3d pixelDirectionU,
                 Eigen::Vector3d pixelDirectionV,
                 float energy,
                 float stepLength,
                 uintptr_t resultPtrDevice) const {
        _mc.project(sourcePosition,
                    detectorOrigin,
                    pixelDirectionU,
                    pixelDirectionV,
                    energy,
                    stepLength,
                    reinterpret_cast<float*>(resultPtrDevice));
    }

  private:
    gpumci::ForwardProjector _mc;
};
class DoseMCPython {
  public:
    DoseMCPython(Eigen::Vector3i volumeSize,
                 Eigen::Vector3d volumeOrigin,
                 Eigen::Vector3d voxelSize,
                 Eigen::Vector2i detectorSize,
                 int n_runs,
                 gpumci::MaterialData data,
                 gpumci::InteractionTables rayleighTables,
                 gpumci::InteractionTables comptonTables)
        : _mc(volumeSize,
              volumeOrigin,
              voxelSize,
              detectorSize,
              n_runs,
              data,
              rayleighTables,
              comptonTables) {
    }

    void setData(uintptr_t densityDevice,
                 uintptr_t materialTypeDevice) {
        _mc.setData(reinterpret_cast<const float*>(densityDevice),
                    reinterpret_cast<const uint8_t*>(materialTypeDevice));
    }

    void project(Eigen::Vector3d sourcePosition,
                 Eigen::Vector3d detectorOrigin,
                 Eigen::Vector3d pixelDirectionU,
                 Eigen::Vector3d pixelDirectionV,
                 uintptr_t resultPrimaryPtrDevice,
                 uintptr_t resultScatterPtrDevice,
                 uintptr_t resultDoseVolumePtrDevice) const {
        _mc.project(sourcePosition,
                    detectorOrigin,
                    pixelDirectionU,
                    pixelDirectionV,
                    reinterpret_cast<float*>(resultPrimaryPtrDevice),
                    reinterpret_cast<float*>(resultScatterPtrDevice),
                    reinterpret_cast<float*>(resultDoseVolumePtrDevice));
    }

  private:
    gpumci::DoseMC _mc;
};

class PhaseSpaceMCPython {
  public:
    PhaseSpaceMCPython(Eigen::Vector3i volumeSize,
                       Eigen::Vector3d volumeOrigin,
                       Eigen::Vector3d voxelSize,
                       Eigen::Vector2i detectorSize,
                       int n_runs,
                       gpumci::MaterialData data,
                       gpumci::InteractionTables rayleighTables,
                       gpumci::InteractionTables comptonTables)
        : _mc(volumeSize,
              volumeOrigin,
              voxelSize,
              detectorSize,
              n_runs,
              data,
              rayleighTables,
              comptonTables) {
    }

    void setData(uintptr_t densityDevice,
                 uintptr_t materialTypeDevice) {
        _mc.setData(reinterpret_cast<const float*>(densityDevice),
                    reinterpret_cast<const uint8_t*>(materialTypeDevice));
    }

    void project(Eigen::Vector3d sourcePosition,
        Eigen::Vector3d detectorOrigin,
        Eigen::Vector3d pixelDirectionU,
        Eigen::Vector3d pixelDirectionV,
        std::vector<gpumci::cuda::CudaMonteCarloParticle> particles,
        uintptr_t resultPrimaryPtrDevice,
        uintptr_t resultScatterPtrDevice) const {
        _mc.project(sourcePosition,
            detectorOrigin,
            pixelDirectionU,
            pixelDirectionV,
            particles,
            reinterpret_cast<float*>(resultPrimaryPtrDevice),
            reinterpret_cast<float*>(resultScatterPtrDevice));
    }

  private:
    gpumci::PhaseSpaceMC _mc;
};


class PhaseSpaceStorePhotonsMCPython {
public:
    PhaseSpaceStorePhotonsMCPython(Eigen::Vector3i volumeSize,
                                   Eigen::Vector3d volumeOrigin,
                                   Eigen::Vector3d voxelSize,
                                   gpumci::MaterialData data,
                                   gpumci::InteractionTables rayleighTables,
                                   gpumci::InteractionTables comptonTables)
        : _mc(volumeSize,
              volumeOrigin,
              voxelSize,
              data,
              rayleighTables,
              comptonTables) {
    }

    void setData(uintptr_t densityDevice,
                 uintptr_t materialTypeDevice) {
        _mc.setData(reinterpret_cast<const float*>(densityDevice),
                    reinterpret_cast<const uint8_t*>(materialTypeDevice));
    }

    void project(const std::vector<gpumci::cuda::CudaMonteCarloParticle>& particles_in,
                 std::vector<gpumci::cuda::CudaMonteCarloParticle>& particles_out) const {
        _mc.project(particles_in, particles_out);
    }

private:
    gpumci::PhaseSpaceStorePhotonsMC _mc;
};

class GainMCPython {
  public:
    GainMCPython(Eigen::Vector3i volumeSize,
                 Eigen::Vector3d volumeOrigin,
                 Eigen::Vector3d voxelSize,
                 Eigen::Vector2i detectorSize,
                 gpumci::MaterialData data,
                 gpumci::InteractionTables rayleighTables,
                 gpumci::InteractionTables comptonTables)
        : _mc(volumeSize,
              volumeOrigin,
              voxelSize,
              detectorSize,
              data,
              rayleighTables,
              comptonTables) {
    }

    void setData(uintptr_t densityDevice,
                 uintptr_t materialTypeDevice) {
        _mc.setData(reinterpret_cast<const float*>(densityDevice),
                    reinterpret_cast<const uint8_t*>(materialTypeDevice));
    }

    void project(Eigen::Vector3d sourcePosition,
                 Eigen::Vector3d detectorOrigin,
                 Eigen::Vector3d pixelDirectionU,
                 Eigen::Vector3d pixelDirectionV,
                 uintptr_t resultPrimaryPtrDevice,
                 uintptr_t resultScatterPtrDevice,
                 uintptr_t gainPtrDevice,
                 float energyMin,
                 float energyMax) const {
        _mc.project(sourcePosition,
                    detectorOrigin,
                    pixelDirectionU,
                    pixelDirectionV,
                    reinterpret_cast<float*>(resultPrimaryPtrDevice),
                    reinterpret_cast<float*>(resultScatterPtrDevice),
                    reinterpret_cast<float*>(gainPtrDevice),
                    energyMin,
                    energyMax);
    }

  private:
    gpumci::GainMC _mc;
};

class GainDoseMCPython {
public:
    GainDoseMCPython(Eigen::Vector3i volumeSize,
                     Eigen::Vector3d volumeOrigin,
                     Eigen::Vector3d voxelSize,
                     Eigen::Vector2i detectorSize,
                     gpumci::MaterialData data,
                     gpumci::InteractionTables rayleighTables,
                     gpumci::InteractionTables comptonTables)
        : _mc(volumeSize,
              volumeOrigin,
              voxelSize,
              detectorSize,
              data,
              rayleighTables,
              comptonTables) {
    }

    void setData(uintptr_t densityDevice,
                 uintptr_t materialTypeDevice) {
        _mc.setData(reinterpret_cast<const float*>(densityDevice),
                    reinterpret_cast<const uint8_t*>(materialTypeDevice));
    }

    void project(Eigen::Vector3d sourcePosition,
                 Eigen::Vector3d detectorOrigin,
                 Eigen::Vector3d pixelDirectionU,
                 Eigen::Vector3d pixelDirectionV,
                 uintptr_t resultDosePtrDevice,
                 uintptr_t gainPtrDevice,
                 float energyMin,
                 float energyMax) const {
        _mc.project(sourcePosition,
                    detectorOrigin,
                    pixelDirectionU,
                    pixelDirectionV,
                    reinterpret_cast<float*>(resultDosePtrDevice),
                    reinterpret_cast<float*>(gainPtrDevice),
                    energyMin,
                    energyMax);
    }

private:
    gpumci::GainDoseMC _mc;
};

class GainMCSimplePython {
  public:
    GainMCSimplePython(Eigen::Vector3i volumeSize,
                       Eigen::Vector3d volumeOrigin,
                       Eigen::Vector3d voxelSize,
                       Eigen::Vector2i detectorSize,
                       gpumci::MaterialData data,
                       gpumci::InteractionTables rayleighTables,
                       gpumci::InteractionTables comptonTables)
        : _mc(volumeSize,
              volumeOrigin,
              voxelSize,
              detectorSize,
              data,
              rayleighTables,
              comptonTables),
          _density(volumeSize.prod()),
          _materialType(volumeSize.prod()),
          _resultPrimary(detectorSize.prod()),
          _resultScatter(detectorSize.prod()),
          _gain(detectorSize.prod()) {
    }

    void setData(numeric::array density,
                 numeric::array materialType) {
        if (!PyArray_ISFARRAY((PyArrayObject*)density.ptr()))
            throw std::runtime_error("density has to be F order");

        if (!PyArray_ISFARRAY((PyArrayObject*)materialType.ptr()))
            throw std::runtime_error("material has to be F order");

        _density.copy_from_host(getDataPtr<float>(density));
        _materialType.copy_from_host(getDataPtr<uint8_t>(materialType));

        _mc.setData(_density.device_ptr(),
                    _materialType.device_ptr());
    }

    void project(Eigen::Vector3d sourcePosition,
                 Eigen::Vector3d detectorOrigin,
                 Eigen::Vector3d pixelDirectionU,
                 Eigen::Vector3d pixelDirectionV,
                 numeric::array resultPrimary,
                 numeric::array resultScatter,
                 numeric::array gain,
                 float energyMin,
                 float energyMax) const {
        if (!PyArray_ISFARRAY((PyArrayObject*)resultPrimary.ptr()))
            throw std::runtime_error("primary has to be F order");

        if (!PyArray_ISFARRAY((PyArrayObject*)resultScatter.ptr()))
            throw std::runtime_error("scatter has to be F order");

        if (!PyArray_ISFARRAY((PyArrayObject*)gain.ptr()))
            throw std::runtime_error("gain has to be F order");

        _resultPrimary.copy_from_host(getDataPtr<float>(resultPrimary));
        _resultScatter.copy_from_host(getDataPtr<float>(resultScatter));
        _gain.copy_from_host(getDataPtr<float>(gain));

        _mc.project(sourcePosition,
                    detectorOrigin,
                    pixelDirectionU,
                    pixelDirectionV,
                    _resultPrimary.device_ptr(),
                    _resultScatter.device_ptr(),
                    _gain.device_ptr(),
                    energyMin,
                    energyMax);

        _resultPrimary.copy_to_host(getDataPtr<float>(resultPrimary));
        _resultScatter.copy_to_host(getDataPtr<float>(resultScatter));
        _gain.copy_to_host(getDataPtr<float>(gain));
    }

  private:
    CudaMemory<float> _density;
    CudaMemory<uint8_t> _materialType;
    mutable CudaMemory<float> _resultPrimary;
    mutable CudaMemory<float> _resultScatter;
    mutable CudaMemory<float> _gain;
    gpumci::GainMC _mc;
};
class SpectMCPython {
  public:
    SpectMCPython(Eigen::Vector3i volumeSize,
                  Eigen::Vector3d volumeOrigin,
                  Eigen::Vector3d voxelSize,
                  Eigen::Vector2i detectorSize,
                  int n_runs,
                  gpumci::MaterialData data,
                  gpumci::InteractionTables rayleighTables,
                  gpumci::InteractionTables comptonTables)
        : _mc(volumeSize,
              volumeOrigin,
              voxelSize,
              detectorSize,
              n_runs,
              data,
              rayleighTables,
              comptonTables) {
    }

    void setData(uintptr_t densityDevice,
                 uintptr_t materialTypeDevice) {
        _mc.setData(reinterpret_cast<const float*>(densityDevice),
                    reinterpret_cast<const uint8_t*>(materialTypeDevice));
    }

    void project(Eigen::Vector3d sourcePosition,
                 Eigen::Vector3d detectorOrigin,
                 Eigen::Vector3d pixelDirectionU,
                 Eigen::Vector3d pixelDirectionV,
                 uintptr_t resultPrimaryPtrDevice,
                 uintptr_t resultScatterPtrDevice,
                 uintptr_t resultDoseVolumePtrDevice) const {
        _mc.project(sourcePosition,
                    detectorOrigin,
                    pixelDirectionU,
                    pixelDirectionV,
                    reinterpret_cast<float*>(resultPrimaryPtrDevice),
                    reinterpret_cast<float*>(resultScatterPtrDevice),
                    reinterpret_cast<float*>(resultDoseVolumePtrDevice));
    }

  private:
    gpumci::SpectMC _mc;
};

class CudaCosWeighting {
  public:
    CudaCosWeighting(Eigen::Vector2i detectorSize) : cosweighting(detectorSize) {
    }

    void apply(Eigen::Vector3d sourcePosition,
               Eigen::Vector3d detectorOrigin,
               Eigen::Vector3d pixelDirectionU,
               Eigen::Vector3d pixelDirectionV,
               uintptr_t sourcePtr,
               uintptr_t targetPtr) {
        cosweighting.apply(sourcePosition,
                           detectorOrigin,
                           pixelDirectionU,
                           pixelDirectionV,
                           reinterpret_cast<float*>(sourcePtr),
                           reinterpret_cast<float*>(targetPtr));
    }

  private:
    gpumci::CudaCosWeighting cosweighting;
};

std::vector<gpumci::cuda::CudaMonteCarloParticle> make_particle_array(Eigen::ArrayXXd positions,
                                                                      Eigen::ArrayXXd directions,
                                                                      Eigen::ArrayXd energies,
                                                                      Eigen::ArrayXd weights) {

    if (positions.cols() != 3) {
        throw std::runtime_error("positions needs to have 3 cols");
    }
    if (directions.cols() != 3) {
        throw std::runtime_error("directions needs to have 3 cols");
    }

    size_t n_particles = positions.rows();

    if (directions.rows() != n_particles ||
        energies.size() != n_particles ||
        weights.size() != n_particles) {
        throw std::runtime_error("arrays not same length");
    }

    std::vector<gpumci::cuda::CudaMonteCarloParticle> array(n_particles);

    for (size_t i = 0; i < n_particles; i++) {
        array[i] = gpumci::cuda::CudaMonteCarloParticle(positions.row(i),
                                                        directions.row(i),
                                                        energies[i],
                                                        1.0);
    }
    return array;
}

template <class T>
class no_compare_indexing_suite : public vector_indexing_suite<T, false, no_compare_indexing_suite<T>> {
  public:
    static bool contains(T& container, typename T::value_type const& key) {
        throw std::logic_error("containment checking not supported on this container");
    }
};

numeric::array get_position(const gpumci::cuda::CudaMonteCarloParticle& particle){
    numeric::array position = makeArray<float>(3);
    position[0] = particle.position.x;
    position[1] = particle.position.y;
    position[2] = particle.position.z;
    return position;
}

numeric::array get_direction(const gpumci::cuda::CudaMonteCarloParticle& particle){
    numeric::array direction = makeArray<float>(3);
    direction[0] = particle.direction.x;
    direction[1] = particle.direction.y;
    direction[2] = particle.direction.z;
    return direction;
}

// Expose classes and methods to Python
BOOST_PYTHON_MODULE(GPUMCIPy) {
    auto result = _import_array(); //Import numpy
    if (result < 0) {
        PyErr_Print();
        PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import");
        return;
    }

    boost::python::numeric::array::set_module_and_type("numpy", "ndarray");

    export_eigen_conv();

    class_<AbsorbingMCPython>("AbsorbingMC", "Documentation",
                              init<Eigen::Vector3i, Eigen::Vector3d, Eigen::Vector3d, Eigen::Vector2i>())
        .def("setData", &AbsorbingMCPython::setData)
        .def("project", &AbsorbingMCPython::project);

    class_<SimpleMCPython>("SimpleMC", "Documentation",
                           init<Eigen::Vector3i, Eigen::Vector3d, Eigen::Vector3d, Eigen::Vector2i>())
        .def("setData", &SimpleMCPython::setData)
        .def("project", &SimpleMCPython::project);

    class_<gpumci::MaterialData>("MaterialData", "Documentation",
                                 init<double, Eigen::ArrayXXd, Eigen::ArrayXXd, Eigen::ArrayXXd>());

    class_<std::vector<Eigen::ArrayXXd>>("vector_array")
        .def(no_compare_indexing_suite<std::vector<Eigen::ArrayXXd>>());

    class_<gpumci::InteractionTables>("InteractionTables", "Documentation",
                                      init<double, std::vector<Eigen::ArrayXXd>>());

    class_<PrecomputedMCPython>("PrecomputedMC", "Documentation",
                                init<Eigen::Vector3i, Eigen::Vector3d, Eigen::Vector3d, Eigen::Vector2i, int, gpumci::MaterialData, gpumci::InteractionTables, gpumci::InteractionTables>())
        .def("setData", &PrecomputedMCPython::setData)
        .def("project", &PrecomputedMCPython::project);

    class_<ForwardProjectorPython>("ForwardProjector", "Documentation",
                                   init<Eigen::Vector3i, Eigen::Vector3d, Eigen::Vector3d, Eigen::Vector2i, gpumci::MaterialData>())
        .def("setData", &ForwardProjectorPython::setData)
        .def("project", &ForwardProjectorPython::project);

    class_<DoseMCPython>("DoseMC", "Documentation",
                         init<Eigen::Vector3i, Eigen::Vector3d, Eigen::Vector3d, Eigen::Vector2i, int, gpumci::MaterialData, gpumci::InteractionTables, gpumci::InteractionTables>())
        .def("setData", &DoseMCPython::setData)
        .def("project", &DoseMCPython::project);

    class_<SpectMCPython>("SpectMC", "Documentation",
                          init<Eigen::Vector3i, Eigen::Vector3d, Eigen::Vector3d, Eigen::Vector2i, int, gpumci::MaterialData, gpumci::InteractionTables, gpumci::InteractionTables>())
        .def("setData", &SpectMCPython::setData)
        .def("project", &SpectMCPython::project);

    class_<gpumci::cuda::CudaMonteCarloParticle>("particle", "Documentation",
                                                 init<Eigen::Vector3d, Eigen::Vector3d, double, double>())
        .add_property("position", get_position)
        .add_property("direction", get_direction)
        .add_property("energy", &gpumci::cuda::CudaMonteCarloParticle::energy)
        .add_property("weight", &gpumci::cuda::CudaMonteCarloParticle::weight);
    def("make_particle_array", &make_particle_array);

    class_<std::vector<gpumci::cuda::CudaMonteCarloParticle>>("particle_array")
        .def(no_compare_indexing_suite<std::vector<gpumci::cuda::CudaMonteCarloParticle>>());

    class_<PhaseSpaceMCPython>("PhaseSpaceMC", "Documentation",
                               init<Eigen::Vector3i, Eigen::Vector3d, Eigen::Vector3d, Eigen::Vector2i, int, gpumci::MaterialData, gpumci::InteractionTables, gpumci::InteractionTables>())
        .def("setData", &PhaseSpaceMCPython::setData)
        .def("project", &PhaseSpaceMCPython::project);

    class_<PhaseSpaceStorePhotonsMCPython>("PhaseSpaceStorePhotonsMC", "Documentation",
        init<Eigen::Vector3i, Eigen::Vector3d, Eigen::Vector3d, gpumci::MaterialData, gpumci::InteractionTables, gpumci::InteractionTables>())
        .def("setData", &PhaseSpaceStorePhotonsMCPython::setData)
        .def("project", &PhaseSpaceStorePhotonsMCPython::project);

    class_<GainMCPython>("GainMC", "Documentation",
                         init<Eigen::Vector3i, Eigen::Vector3d, Eigen::Vector3d, Eigen::Vector2i, gpumci::MaterialData, gpumci::InteractionTables, gpumci::InteractionTables>())
        .def("setData", &GainMCPython::setData)
        .def("project", &GainMCPython::project);    
    
    class_<GainDoseMCPython>("GainDoseMC", "Documentation",
                         init<Eigen::Vector3i, Eigen::Vector3d, Eigen::Vector3d, Eigen::Vector2i, gpumci::MaterialData, gpumci::InteractionTables, gpumci::InteractionTables>())
        .def("setData", &GainDoseMCPython::setData)
        .def("project", &GainDoseMCPython::project);

    class_<GainMCSimplePython>("GainMCSimple", "Documentation",
                               init<Eigen::Vector3i, Eigen::Vector3d, Eigen::Vector3d, Eigen::Vector2i, gpumci::MaterialData, gpumci::InteractionTables, gpumci::InteractionTables>())
        .def("setData", &GainMCSimplePython::setData)
        .def("project", &GainMCSimplePython::project);
}
