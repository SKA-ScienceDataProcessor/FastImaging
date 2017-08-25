/**
* @file matstp.h
* @brief MatStp matrix class.
*/

#ifndef MATSTP_H
#define MATSTP_H

#include "../types.h"
#include <armadillo>
#include <memory>

#define CACHE_LINE_SIZE 64

namespace stp {

/**
 * @brief The ZeroMemAlloc buffer class
 *
 * Creates zeroed buffer using calloc function.
 */
template <typename T>
class ZeroMemAlloc {
public:
    /**
     * @brief ZeroMemAlloc default constructor
     */
    ZeroMemAlloc()
        : mem_ptr()
        , num_elems(0)
    {
    }

    /**
     * @brief ZeroMemAlloc constructor using given length
     */
    ZeroMemAlloc(size_t length)
        : mem_ptr(std::unique_ptr<T, std::function<void(T*)>>((T*)std::calloc(length, sizeof(T)), [](T* ptr) {
            if (ptr != nullptr) {
                std::free(ptr);
                ptr = nullptr;
            }
        }))
        , num_elems(length)
    {
    }

    // Delete copy and assignment constructors
    ZeroMemAlloc(ZeroMemAlloc const&) = delete;
    ZeroMemAlloc& operator=(ZeroMemAlloc const&) = delete;

    /**
     * @brief ZeroMemAlloc move constructor
     */
    ZeroMemAlloc(ZeroMemAlloc&& other)
        : mem_ptr(std::move(other.mem_ptr))
        , num_elems(other.num_elems)
    {
    }

    /**
     * @brief ZeroMemAlloc move assignment operator
     */
    ZeroMemAlloc& operator=(ZeroMemAlloc&& other)
    {
        // Self-assignment detection
        if (this == &other)
            return *this;

        mem_ptr = std::move(other.mem_ptr);
        num_elems = other.num_elems;

        return *this;
    }

    /**
     * Smart pointer for the memory buffer
     */
    std::unique_ptr<T, std::function<void(T*)>> mem_ptr;
    /**
     * Buffer size
     */
    size_t num_elems;
};

/**
 * @brief The MatStp matrix class
 *
 * Creates zeroed matrix that uses ZeroMemAlloc (based on calloc function) and inherints armadillo Mat methods.
 */
template <typename T>
class MatStp : private ZeroMemAlloc<T>, public arma::Mat<T> {
public:
    /**
     * @brief MatStp default constructor
     */
    MatStp() = default;

    /**
     * @brief MatStp constructor that receives matrix dimensions
     */
    MatStp(arma::uword n_rows, arma::uword n_cols)
        : ZeroMemAlloc<T>(n_rows * n_cols + CACHE_LINE_SIZE)
        , arma::Mat<T>(aligned_mem_ptr(), n_rows, n_cols, false, false)
    {
    }

    /**
     * @brief MatStp move constructor
     */
    MatStp(MatStp&& other)
        : ZeroMemAlloc<T>(std::move(static_cast<ZeroMemAlloc<T>&>(other))) // Static_cast is optional, since implicit upcast also works
        , arma::Mat<T>(std::move(static_cast<arma::Mat<T>&>(other)))
    {
    }

    /**
     * @brief MatStp move assignment operator
     */
    MatStp<T>& operator=(MatStp&& other)
    {
        // Self-assignment detection
        if (&other == this)
            return *this;

        ZeroMemAlloc<T>::operator=(std::move(static_cast<ZeroMemAlloc<T>&>(other))); // Static_cast is optional, since implicit upcast also works
        arma::Mat<T>::operator=(std::move(static_cast<arma::Mat<T>&>(other)));
        return *this;
    }

    /**
     * @brief MatStp sum assignment operator
     */
    MatStp<T>& operator+=(MatStp& other)
    {
        arma::Mat<T>::operator+=(other);
        return *this;
    }

    /**
     * @brief Delete matrix buffer
     */
    void delete_matrix_buffer()
    {
        ZeroMemAlloc<T>::mem_ptr.reset();
    }

private:
    /**
     * @brief Get aligned memory position
     */
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
