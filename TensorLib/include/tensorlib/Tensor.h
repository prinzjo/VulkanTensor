#pragma once

#include <array>
#include <cassert>
#include <iostream>
#include <ostream>
#include <string>
#include <vector>

namespace tl
{
    template <typename T>
    class Tensor
    {
    public:
        explicit Tensor(std::vector<size_t> shape)
            : m_Shape(std::move(shape))
        {
            if (m_Shape.empty())
            {
                std::cerr << "Tensor: Shape must not be empty\n";
                return;
            }

            for (size_t dim : m_Shape)
            {
                if (dim == 0)
                {
                    std::cerr << "Tensor: dimensions must be non-zero\n";
                    return;
                }
            }

            m_Strides.resize(m_Shape.size());

            size_t stride = 1;
            for (size_t i = m_Shape.size(); i-- > 0; )
            {
                m_Strides[i] = stride;
                stride *= m_Shape[i];
            }

            m_Data.resize(stride);
        }

        Tensor(std::vector<size_t> shape, const T& fillValue)
            : Tensor(std::move(shape))
        {
            Fill(fillValue);
        }

        size_t Rank() const { return m_Shape.size(); }
        size_t Size() const { return m_Data.size(); }

        const std::vector<size_t>& Shape() const { return m_Shape; }
        const std::vector<size_t>& Strides() const { return m_Strides; }

        T* Data() { return m_Data.data(); }
        const T* Data() const { return m_Data.data(); }

        void Fill(const T& value)
        {
            std::fill(m_Data.begin(), m_Data.end(), value);
        }

        template <typename... Indices>
        T& operator()(Indices... indices)
        {
            constexpr size_t N = sizeof...(Indices);
            std::array<size_t, N> idx { static_cast<size_t>(indices)... };
            return m_Data[LinearIndex(idx)];
        }

        template <typename... Indices>
        const T& operator()(Indices... indices) const
        {
            constexpr size_t N = sizeof...(Indices);
            std::array<size_t, N> idx { static_cast<size_t>(indices)... };
            return m_Data[LinearIndex(idx)];
        }

        T& At(const std::vector<size_t>& indices)
        {
            return m_Data[CheckedLinearIndex(indices)];
        }

        const T& At(const std::vector<size_t>& indices) const
        {
            return m_Data[CheckedLinearIndex(indices)];
        }

        bool operator==(const Tensor& other) const
        {
            return m_Shape == other.m_Shape && m_Data == other.m_Data;
        }

        bool operator!=(const Tensor& other) const
        {
            return !(*this == other);
        }

        friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor)
        {
            os << "Tensor(rank=" << tensor.Rank() << ", shape=[";

            for (size_t i = 0; i < tensor.m_Shape.size(); i++)
            {
                os << tensor.m_Shape[i];
                if (i + 1 != tensor.m_Shape.size())
                    os << ", ";
            }

            os << "])\n";

            std::vector<size_t> idx(tensor.Rank(), 0);
            tensor.PrintRecursive(os, idx, 0, 0);
            os << "\n";

            return os;
        }

    private:
        template <size_t N>
        size_t LinearIndex(const std::array<size_t, N>& idx) const
        {
            assert(N == m_Shape.size() && "Tensor: index count must match rank");

            size_t linear = 0;
            for (size_t i = 0; i < N; i++)
            {
                assert(idx[i] < m_Shape[i] && "Tensor: index out of bounds");
                linear += idx[i] * m_Strides[i];
            }

            return linear;
        }

        [[nodiscard]] size_t CheckedLinearIndex(const std::vector<size_t>& idx) const
        {
            if (idx.size() != m_Shape.size())
            {
                std::cerr << "Tensor::At: index count does not match rank\n";
                return 0;
            }

            size_t linear = 0;
            for (size_t i = 0; i < idx.size(); i++)
            {
                if (idx[i] >= m_Shape[i])
                {
                    std::cerr << "Tensor::At: index out of bounds\n";
                    return 0;
                }

                linear += idx[i] * m_Strides[i];
            }

            return linear;
        }

        void PrintRecursive(std::ostream& os, std::vector<size_t>& idx, size_t dim, size_t depth) const
        {
            const std::string indent(depth * 2, ' ');

            if (dim == m_Shape.size() - 1)
            {
                os << "[";
                for (size_t i = 0; i < m_Shape[dim]; i++)
                {
                    idx[dim] = i;
                    os << At(idx);

                    if (i + 1 != m_Shape[dim])
                        os << " ";
                }
                os << "]";
                return;
            }

            os << "[\n";
            for (size_t i = 0; i < m_Shape[dim]; i++)
            {
                idx[dim] = i;
                os << indent << "  ";
                PrintRecursive(os, idx, dim + 1, depth + 1);

                if (i + 1 != m_Shape[dim])
                    os << ",";

                os << "\n";
            }
            os << indent << "]";
        }

        std::vector<size_t> m_Shape;
        std::vector<size_t> m_Strides;
        std::vector<T> m_Data;
    };
}