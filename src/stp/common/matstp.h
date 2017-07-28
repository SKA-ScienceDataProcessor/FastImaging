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
        mem_ptr = std::unique_ptr<T, std::function<void(T*)>>((T*)std::calloc(length, sizeof(T)), [](T* ptr) {
            if (ptr != nullptr) {
                std::free(ptr);
                ptr = nullptr;
            }
        });
        num_elems = length;
    }

    // Delete copy and assignment constructors
    ZeroMemAlloc(ZeroMemAlloc const&) = delete;
    ZeroMemAlloc& operator=(ZeroMemAlloc const&) = delete;

    // Define move constructor
    ZeroMemAlloc(ZeroMemAlloc&& other)
        : mem_ptr(std::move(other.mem_ptr))
    {
        other.mem_ptr = nullptr;
    }

    // Define move assignment operator
    ZeroMemAlloc& operator=(ZeroMemAlloc&& other)
    {
        // Self-assignment detection
        if (&other == this)
            return *this;

        mem_ptr = std::move(other.mem_ptr);
        num_elems = other.num_elems;

        return *this;
    }

    std::unique_ptr<T, std::function<void(T*)>> mem_ptr;
    size_t num_elems;
};

template <typename T>
class MatStp : private ZeroMemAlloc<T>, public arma::Mat<T> {
public:
    // Default constructor
    MatStp() = default;

    // Constructor
    MatStp(arma::uword n_rows, arma::uword n_cols)
        : ZeroMemAlloc<T>(n_rows * n_cols + CACHE_LINE_SIZE)
        , arma::Mat<T>(aligned_mem_ptr(), n_rows, n_cols, false, false)
    {
    }

    // Move constructor
    MatStp(MatStp&& other)
        : ZeroMemAlloc<T>(std::move(other))
        , arma::Mat<T>(std::move(other))
    {
    }

    // Move assignment operator
    MatStp<T>& operator=(MatStp&& other)
    {
        // Self-assignment detection
        if (&other == this)
            return *this;

        ZeroMemAlloc<T>::operator=(std::move(other));
        arma::Mat<T>::operator=(std::move(other));
        return *this;
    }

    // Sum assignment operator
    MatStp<T>& operator+=(MatStp& other)
    {
        arma::Mat<T>::operator+=(other);
        return *this;
    }

    // Deletes matrix buffer
    void delete_matrix_buffer()
    {
        ZeroMemAlloc<T>::mem_ptr.reset();
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
