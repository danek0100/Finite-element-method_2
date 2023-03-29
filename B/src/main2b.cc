// Include files
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>

#include "FEM2b.h"
#include "writeSolutions.h"

using namespace dealii;

int main ()
{
	try
	{
 		deallog.depth_console (0);
		const int dimension = 3;
    	FEM<dimension> problemObject;

		std::vector<unsigned int> num_of_elems( dimension );
		num_of_elems[0] = 8;
		num_of_elems[1] = 16;
		num_of_elems[2] = 4;

		problemObject.generate_mesh(num_of_elems);
		problemObject.setup_system();
		problemObject.assemble_system();
		problemObject.solve();
		problemObject.output_results();
    
    	char tag[21];
    	sprintf( tag, "CA2b" );
    	writeSolutionsToFileCA2( problemObject.D, tag );
  }
  catch ( std::exception &exc )
  {
    std::cerr << std::endl << std::endl
	      << "----------------------------------------------------"
	      << std::endl;
    std::cerr << "Exception on processing: " << std::endl
	      << exc.what() << std::endl
	      << "Aborting!" << std::endl
	      << "----------------------------------------------------"
	      << std::endl;

    return 1;
  }
  catch (...)
  {
    std::cerr << std::endl << std::endl
	      << "----------------------------------------------------"
	      << std::endl;
    std::cerr << "Unknown exception!" << std::endl
	      << "Aborting!" << std::endl
	      << "----------------------------------------------------"
	      << std::endl;
    return 1;
  }

  return 0;
}
