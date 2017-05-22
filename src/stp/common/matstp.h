#ifndef MATSTP_H
#define MATSTP_H

#include "../types.h"
#include <armadillo>
#include <memory>

#define CACHE_LINE_SIZE 64

namespace stp {

template <typename T>
class ZeroMemAlloc {
public:
    // Default constructor
    ZeroMemAlloc()
        : mem_ptr()
        , num_elems(0)
    {
    }

    // Constructor
    ZeroMemAlloc(size_t length)
    {
        mem_ptr = std::unique_ptr<T, std::function<void(T*)> >((T*)std::calloc(length, sizeof(T)), [](T* ptr) {
            if (ptr != nullptr) {
                std::free(ptr);
            }
        });
        num_elems = length;
    }

    // Delete copy and assign constructors
    ZeroMemAlloc(ZeroMemAlloc const&) = delete;
    ZeroMemAlloc& operator=(ZeroMemAlloc const&) = delete;

    // Define move constructor
    ZeroMemAlloc(ZeroMemAlloc&& other)
        : mem_ptr(std::move(other.mem_ptr))
    {
    }

    size_t num_elems;
    std::unique_ptr<T, std::function<void(T*)> > mem_ptr;
};

template <typename T>
class MatStp : private ZeroMemAlloc<T>, public arma::Mat<T> {
public:
    // Default constructor
    MatStp()
        : ZeroMemAlloc<T>()
        , arma::Mat<T>()
    {
    }

    // Constructor
    MatStp(arma::uword n_rows, arma::uword n_cols)
        : ZeroMemAlloc<T>(n_rows * n_cols + CACHE_LINE_SIZE)
        , arma::Mat<T>(aligned_mem_ptr(), n_rows, n_cols, false, true)
    {
    }

    // Move constructor
    MatStp(MatStp&& other)
        : ZeroMemAlloc<T>(std::move(other))
        , arma::Mat<T>(std::move(other))
    {
    }

private:
    // Get aligned memory position
    T* aligned_mem_ptr()
    {
        size_t length = ZeroMemAlloc<T>::num_elems;
        void* p = ZeroMemAlloc<T>::mem_ptr.get();
        std::align(CACHE_LINE_SIZE, sizeof(T), p, length);

        return (T*)p;
    }
};
}

#endif /* MATSTP_H */
