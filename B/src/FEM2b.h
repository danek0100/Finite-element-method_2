#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_q.h>

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <math.h>

using namespace dealii;

template < int dim >
class FEM
{
  public:
  FEM();
  ~FEM();

  double basis_function( unsigned int node, double xi_1, double xi_2, double xi_3 );
  std::vector< double > basis_gradient( unsigned int node, double xi_1, double xi_2, double xi_3 );

  void generate_mesh( std::vector< unsigned int > numberOfElements );
  void define_boundary_conds();
  void setup_system();
  void assemble_system();
  void solve();
  void output_results();

  Triangulation< dim >   triangulation;
  FESystem< dim >        fe;
  DoFHandler< dim >      dof_handler;

  unsigned int	      quadRule;
  std::vector<double> quad_points;
  std::vector<double> quad_weight;

  SparsityPattern    	             sparsity_pattern;
  SparseMatrix< double >	         K;
  Vector< double >       	         D, F;
  Table< 2, double >	             nodeLocation;
  std::map< unsigned int, double > boundary_values;
  double x_min, x_max, y_min, y_max, z_min, z_max;

  std::vector< std::string > nodal_solution_names;
  std::vector< DataComponentInterpretation::DataComponentInterpretation > nodal_data_component_interpretation;
};

template < int dim >
FEM< dim >::FEM () :fe( FE_Q< dim >( 1 ), 1 ), dof_handler( triangulation )
{
  nodal_solution_names.push_back( "D" );
  nodal_data_component_interpretation.push_back( DataComponentInterpretation::component_is_scalar );
}

template <int dim>
FEM< dim >::~FEM ()
{
  dof_handler.clear();
}

template <int dim>
double FEM< dim >::basis_function( unsigned int node, double xi_1, double xi_2, double xi_3 )
{
  double value = 0.;

  switch(node) {
    case 0:
      value = (1. - xi_1) * (1. - xi_2) * (1 - xi_3) / 8.;
      break;
    case 1:
      value = (1. + xi_1) * (1. - xi_2) * (1 - xi_3) / 8.;
      break;
    case 2:
      value = (1. - xi_1) * (1. + xi_2) * (1 - xi_3) / 8.;
      break;
    case 3:
      value = (1. + xi_1) * (1. + xi_2) * (1 - xi_3) / 8.;
      break;
    case 4:
      value = (1. - xi_1) * (1. - xi_2) * (1 + xi_3) / 8.;
      break;
    case 5:
      value = (1. + xi_1) * (1. - xi_2) * (1 + xi_3) / 8.;
      break;
    case 6:
      value = (1. - xi_1) * (1. + xi_2) * (1 + xi_3) / 8.;
      break;
    case 7:
      value = (1. + xi_1) * (1. + xi_2) * (1 + xi_3) / 8.;
      break;
    default:
      std::cerr << "Error: Invalid node index in basis_function." << std::endl;
      break;
  }

  return value;
}

template < int dim >
std::vector< double > FEM<dim>::basis_gradient( unsigned int node, double xi_1, double xi_2, double xi_3 )
{
  std::vector<double> values( dim, 0.0 );

    switch (node) {
    case 0:
      values[0] = (-1.) * (1. - xi_2) * (1 - xi_3) / 8.;
      values[1] = (1. - xi_1) * (-1.) * (1 - xi_3) / 8.;
      values[2] = -(1. - xi_1) * (1. - xi_2) / 8.;
      break;
    case 1:
      values[0] = (1. - xi_2) * (1 - xi_3) / 8.;
      values[1] = -(1. + xi_1) * (1 - xi_3) / 8.;
      values[2] = - (1. + xi_1) * (1. - xi_2) / 8.;
      break;
    case 2:
      values[0] = -(1. + xi_2) * (1 - xi_3) / 8.;
      values[1] = (1. - xi_1) * (1 - xi_3) / 8.;
      values[2] = - (1. - xi_1) * (1. + xi_2) / 8.;
      break;
    case 3:
      values[0] = (1. + xi_2) * (1 - xi_3) / 8.;
      values[1] = (1. + xi_1) * (1 - xi_3) / 8.;
      values[2] = -(1. + xi_1) * (1. + xi_2) / 8.;
      break;
    case 4:
      values[0] = (-1.) * (1. - xi_2) * (1 + xi_3) / 8.;
      values[1] = (1. - xi_1) * (-1.) * (1 + xi_3) / 8.;
      values[2] = (1. - xi_1) * (1. - xi_2) / 8.;
      break;
    case 5:
      values[0] = (1. - xi_2) * (1 + xi_3) / 8.;
      values[1] = -(1. + xi_1) * (1 + xi_3) / 8.;
      values[2] = (1. + xi_1) * (1. - xi_2) / 8.;
      break;
    case 6:
      values[0] = -(1. + xi_2) * (1 + xi_3) / 8.;
      values[1] = (1. - xi_1) * (1 + xi_3) / 8.;
      values[2] = (1. - xi_1) * (1. + xi_2) / 8.;
      break;
    case 7:
      values[0] = (1. + xi_2) * (1 + xi_3) / 8.;
      values[1] = (1. + xi_1) * (1 + xi_3) / 8.;
      values[2] = (1. + xi_1) * (1. + xi_2) / 8.;
      break;
    default:
      std::cerr << "Invalid node value in basis_gradient()" << std::endl;
      break;
  }

  return values;
}

template <int dim>
void FEM<dim>::generate_mesh( std::vector< unsigned int > numberOfElements )
{
  x_min = 0., x_max = 0.04,
  y_min = 0., y_max = 0.08,
  z_min = 0., z_max = 0.02;

  Point< dim, double > min( x_min, y_min, z_min ), max( x_max, y_max, z_max );
  GridGenerator::subdivided_hyper_rectangle( triangulation, numberOfElements, min, max );
}

template <class T, class U>
bool areAlmostEqual(T a, U b)
{
    constexpr double eps = 1.e-8;
    return std::abs( a - b ) < eps;
}

template < int dim >
void FEM<dim>::define_boundary_conds()
{
  const unsigned int totalNodes = dof_handler.n_dofs();

  for (uint i = 0; i < totalNodes ; ++i) 
  {
    if ( areAlmostEqual( nodeLocation[i][0], x_min ) )
    {
      boundary_values[i] = 300 * (1. + 1./3. * ( nodeLocation[i][1] + nodeLocation[i][2] ) );
    }

    if ( areAlmostEqual( nodeLocation[i][0], x_max ) )
    {
      boundary_values[i] = 310 * ( 1. + 1./3. * ( nodeLocation[i][1] + nodeLocation[i][2] ) );
    }
  }
}

template < int dim >
void FEM<dim>::setup_system()
{
  dof_handler.distribute_dofs( fe );

  MappingQ1< dim, dim > mapping;
  std::vector< Point< dim, double > > dof_coords( dof_handler.n_dofs() );
  nodeLocation.reinit( dof_handler.n_dofs(), dim );
  DoFTools::map_dofs_to_support_points< dim, dim >( mapping, dof_handler, dof_coords );

  for( unsigned int i = 0; i < dof_coords.size(); i++ )
  {
    for( unsigned int j = 0; j < dim; j++ )
    {
      nodeLocation[i][j] = dof_coords[i][j];
    }
  }

  define_boundary_conds();

  sparsity_pattern.reinit( dof_handler.n_dofs(), dof_handler.n_dofs(), dof_handler.max_couplings_between_dofs() );
  DoFTools::make_sparsity_pattern( dof_handler, sparsity_pattern );
  sparsity_pattern.compress();
  K.reinit( sparsity_pattern );
  F.reinit( dof_handler.n_dofs() );
  D.reinit( dof_handler.n_dofs() );

  quadRule = 3; // Let the accuracy be higher
  quad_points.resize( quadRule ); 
  quad_weight.resize( quadRule );

  quad_points[0] = 0;
  quad_points[1] = -0.7745966692414834;
  quad_points[2] = 0.7745966692414834;

  quad_weight[0] = 0.88888888;
  quad_weight[1] = 0.555555556;
  quad_weight[2] = 0.5555555556; 
}

template < int dim >
void FEM<dim>::assemble_system()
{
  K=0; F=0;

  const unsigned int  	      dofs_per_elem = fe.dofs_per_cell;
  FullMatrix< double > 	      Klocal( dofs_per_elem, dofs_per_elem );
  Vector< double >      	    Flocal( dofs_per_elem );
  std::vector< unsigned int > local_dof_indices( dofs_per_elem );

  typename DoFHandler<dim>::active_cell_iterator elem = dof_handler.begin_active(), endc = dof_handler.end();
  for ( ; elem != endc; ++elem )
  {
    elem->get_dof_indices( local_dof_indices );

    FullMatrix< double > Jacobian( dim, dim );
    double detJ;

    Flocal = 0.;
    for( unsigned int q1 = 0; q1 < quadRule; q1++ )
    {
      for( unsigned int q2 = 0; q2 < quadRule; q2++ )
      {
	      for( unsigned int q3 = 0; q3 < quadRule; q3++ )
        {
	        Jacobian = 0.;
	        for( unsigned int i = 0; i < dim; i++ )
          {
	          for( unsigned int j = 0; j < dim; j++ )
            {
	            for( unsigned int A = 0; A < dofs_per_elem; A++ )
              {
		            Jacobian[i][j] += nodeLocation[ local_dof_indices[A] ][i] * basis_gradient( A, quad_points[q1], quad_points[q2], quad_points[q3] )[j];
	            }
	          }
	        }
	        detJ = Jacobian.determinant();
	        for( unsigned int A = 0; A < dofs_per_elem; A++ ) { }
	      }
      }
    }

    FullMatrix< double > invJacob( dim, dim ), kappa( dim, dim );

    kappa = 0.;
    kappa[0][0] = 385.;
    kappa[1][1] = 385.;
    kappa[2][2] = 385.;

    Klocal = 0.;
    for( unsigned int q1 = 0; q1 < quadRule; q1++ )
    {
      for( unsigned int q2 = 0; q2 < quadRule; q2++ )
      {
	      for( unsigned int q3 = 0; q3 < quadRule; q3++ )
        {
	        Jacobian = 0.;
	        for( unsigned int i = 0; i < dim; i++ )
          {
            for( unsigned int j = 0; j < dim; j++ )
            {
              for( unsigned int A = 0; A < dofs_per_elem; A++ )
              {
                Jacobian[i][j] += nodeLocation[ local_dof_indices[A] ][i] * basis_gradient( A, quad_points[q1], quad_points[q2], quad_points[q3] )[j];
              }
            }
	        }
	        detJ = Jacobian.determinant();
          double weight = quad_weight[q1]*quad_weight[q2]*quad_weight[3];
          invJacob.invert( Jacobian );
          for( unsigned int A = 0; A < dofs_per_elem; A++ )
          {
            for( unsigned int B = 0; B < dofs_per_elem; B++ )
            {
              for( unsigned int i = 0; i < dim; i++ )
              {
                for( unsigned int j = 0; j < dim; j++ )
                {
                  for( unsigned int I = 0; I < dim; I++ )
                  {
                    for( unsigned int J = 0; J < dim; J++ )
                    {
                      Klocal[A][B] += basis_gradient( A, quad_points[q1], quad_points[q2], quad_points[q3] )[i] * invJacob[i][I] * kappa[I][J] * basis_gradient( B, quad_points[q1], quad_points[q2], quad_points[q3] )[j] * invJacob[j][J] * weight * detJ;
                    }
                  }
                }
              }
            }
          }
	      }
      }
    }	
    for( unsigned int A = 0; A < dofs_per_elem; A++ )
    {
      for( unsigned int B = 0; B < dofs_per_elem; B++ )
      {
        K.add( local_dof_indices[A], local_dof_indices[B], Klocal[A][B] );
      }
    }
  }
  MatrixTools::apply_boundary_values( boundary_values, K, D, F, false );
}

template <int dim>
void FEM<dim>::solve()
{
  SparseDirectUMFPACK  A;
  A.initialize( K );
  A.vmult( D, F );
}

template< int dim >
void FEM<dim>::output_results ()
{
  std::ofstream output1( "solution.vtk" );
  DataOut< dim > data_out;
  data_out.attach_dof_handler( dof_handler );

  data_out.add_data_vector( D, nodal_solution_names, DataOut<dim>::type_dof_data, nodal_data_component_interpretation );
  data_out.build_patches();
  data_out.write_vtk( output1 );
  output1.close();
}
