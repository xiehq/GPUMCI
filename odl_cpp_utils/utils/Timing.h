#include <chrono>

namespace SimRec2D {
template <typename TimeT = std::chrono::microseconds>
struct Timer {
    Timer(unsigned runs = 1) : _elapsed(0), _runs(runs) {
        assert(runs > 0);
    }

    template <typename F, typename... Args>
    auto run(F func, Args&&... args) -> decltype(func(std::forward<Args>(args)...)) {
        auto start = std::chrono::system_clock::now();

        decltype(func(std::forward<Args>(args)...)) result;

        // Now call the function with all the parameters you need.
        for (unsigned i = 0; i < _runs; i++) {
            result = func(std::forward<Args>(args)...);
        }

        auto duration = std::chrono::duration_cast<TimeT>(std::chrono::system_clock::now() - start);

        _elapsed += duration;

        return result;
    }

    typename TimeT::rep totalTime() const {
        return _elapsed.count();
    }

    void reset() {
        _elapsed = TimeT(0);
    }

  private:
    unsigned _runs;
    TimeT _elapsed;
};
}