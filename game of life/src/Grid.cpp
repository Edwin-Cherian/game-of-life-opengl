#include "Grid.h"
#include <iostream>

Grid::Grid()
{
    cudaMallocManaged( &pgrid, 6000000 );
}

Grid::~Grid()
{
    cudaFree(pgrid);
    std::cout << "make a grid destructor pls";
}

void Grid::AddObject(float* objPtr)
{
	const int cell_col = static_cast<int>(objPtr[0] / (1200.0f / cols));
	const int cell_row = static_cast<int>(objPtr[1] / (700.0f / rows));
	grid[cell_row][cell_col].push_back(objPtr);
}

/* does not remove given object ptr from vector in a grid cell. Instead, it
removes the last element in the vector. This is ok in the given context because every
element will be removed from the grid before being added.
Could be sped up by just clearing vector. consider what to do when other particles
try to clear the same vector.*/
void Grid::RemoveObject(float* objPtr)
{
    const int cell_col = static_cast<int>(objPtr[0] / (1200.0f / cols));
    const int cell_row = static_cast<int>(objPtr[1] / (700.0f / rows));
    //std::cout << "Deleting obj: " << cell_row << " " << cell_col << std::endl;
    grid[cell_row][cell_col].pop_back();
}

std::vector<float*> Grid::FindNear(float* objPtr)
{
    std::vector<float*> nearby;
    const int cell_col = static_cast<int>(objPtr[0] / (1200.0f / cols));
    const int cell_row = static_cast<int>(objPtr[1] / (700.0f / rows));
    for (int i = -1; i < 2; i++)
    {
        for (int j = -1; j < 2; j++)
        {
            if (cell_row + i < 0 || cell_row + i >= rows || cell_col + j < 0 || cell_col + j >= cols)
                continue;
            for (auto v : grid[cell_row + i][cell_col + j])
            {
                nearby.push_back(v);
            }
        }
    }
    return nearby;
}

void Grid::ReadData(float* objPtr)
{
	/*const int cell_row = static_cast<int>(objPtr[0] / rows);
	const int cell_col = static_cast<int>(objPtr[1] / cols);
	grid[rows][cols].push_back(objPtr);*/
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            for (auto v : grid[i][j])
            {
                std::cout << i << " " << j << " " << *v << ", " << *(v+1) << std::endl;
            }
        }
    }
}
