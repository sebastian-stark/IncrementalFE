#include <deal.II/base/exceptions.h>
#include <deal.II/lac/vector.h>
#include <incremental_fe/scalar_functionals/interface_dissipation_00.h>

using namespace dealii;
using namespace std;
using namespace dealii::GalerkinTools;
using namespace incrementalFE;

template<unsigned int spacedim>
InterfaceDissipation00<spacedim>::InterfaceDissipation00(	const vector<DependentField<spacedim-1,spacedim>>	e_sigma,
															const set<types::material_id>						domain_of_integration,
															const Quadrature<spacedim-1>						quadrature,
															GlobalDataIncrementalFE<spacedim>&					global_data,
															const double										d,
															const unsigned int									formulation)
:
ScalarFunctional<spacedim-1, spacedim>(e_sigma, domain_of_integration, quadrature, "InterfaceDissipation00", 1),
global_data(&global_data),
formulation(formulation),
d(d)
{
}

template<unsigned int spacedim>
bool
InterfaceDissipation00<spacedim>::get_h_sigma(	const Vector<double>& 			e_sigma,
												const vector<Vector<double>>&	e_sigma_ref,
												Vector<double>& 				/*hidden_vars*/,
												const Point<spacedim>& 			/*x*/,
												const Tensor<1,spacedim>& 		n,
												double& 						h_sigma,
												Vector<double>& 				h_sigma_1,
												FullMatrix<double>& 			h_sigma_2,
												const tuple<bool, bool, bool>	requested_quantities)
const
{

	Assert(e_sigma.size() == this->e_sigma.size(),ExcMessage("Called get_h_sigma with invalid size of e vector!"));
	Assert(e_sigma_ref.size() >= this->n_ref_sets,ExcMessage("Called get_h_sigma with not enough datasets for the reference values of the independent fields!"));
	Assert(e_sigma_ref[0].size() == this->e_sigma.size(),ExcMessage("Called get_h_sigma with invalid size of e_ref vector!"));

	const double delta_t = global_data->get_t() - global_data->get_t_ref();
	const double nx = n[0];
	const double ny = n[1];
	const double nz = spacedim==3 ? n[2] : 0.0;

	double I_n_I_n_ref;
	if(formulation == 0)
		I_n_I_n_ref=(e_sigma[0]-e_sigma_ref[0][0])*nx + (e_sigma[1]-e_sigma_ref[0][1])*ny + (e_sigma[2]-e_sigma_ref[0][2])*nz;
	else
		I_n_I_n_ref=(e_sigma[0])*nx + (e_sigma[1])*ny + (e_sigma[2])*nz;

	if(get<0>(requested_quantities))
	{
		h_sigma = 0.5/d * I_n_I_n_ref * I_n_I_n_ref/delta_t;
	}
	if(get<1>(requested_quantities))
	{
		h_sigma_1.reinit(3);
		h_sigma_1[0] = 1./d*I_n_I_n_ref*nx/delta_t;
		h_sigma_1[1] = 1./d*I_n_I_n_ref*ny/delta_t;
		h_sigma_1[2] = 1./d*I_n_I_n_ref*nz/delta_t;
	}
	if(get<2>(requested_quantities))
	{
		h_sigma_2.reinit(3,3);
		h_sigma_2(0,0) = 1./d*nx*nx/delta_t;
		h_sigma_2(1,1) = 1./d*ny*ny/delta_t;
		h_sigma_2(2,2) = 1./d*nz*nz/delta_t;
		h_sigma_2(0,1) = h_sigma_2(1,0) = 1./d*nx*ny/delta_t;
		h_sigma_2(1,2) = h_sigma_2(2,1) = 1./d*ny*nz/delta_t;
		h_sigma_2(0,2) = h_sigma_2(2,0) = 1./d*nx*nz/delta_t;
	}
	return false;
}

template class InterfaceDissipation00<2>;
template class InterfaceDissipation00<3>;
