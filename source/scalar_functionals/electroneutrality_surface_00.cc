#include <deal.II/base/exceptions.h>
#include <deal.II/lac/vector.h>

#include <incremental_fe/scalar_functionals/electroneutrality_surface_00.h>

using namespace dealii;
using namespace std;
using namespace dealii::GalerkinTools;
using namespace incrementalFE;

template<unsigned int spacedim>
ElectroNeutralitySurface00<spacedim>::ElectroNeutralitySurface00(	const vector<DependentField<spacedim-1,spacedim>> e_sigma,
																	const set<types::material_id> domain_of_integration,
																	const Quadrature<spacedim-1> quadrature,
																	GlobalDataIncrementalFE<spacedim>& global_data,
																	const double alpha):
ScalarFunctional<spacedim-1, spacedim>(e_sigma, domain_of_integration, quadrature, "ElectroNeutralitySurface00", 1),
global_data(&global_data),
alpha(alpha)
{
	Assert( (alpha >= 0.0) && (alpha <= 1.0), ExcMessage("alpha must be in the range 0 <= alpha <= 1 !"));
}

template<unsigned int spacedim>
bool
ElectroNeutralitySurface00<spacedim>::get_h_sigma(	const Vector<double>& 			e_sigma,
													const vector<Vector<double>>&	e_sigma_ref_sets,
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
	Assert(e_sigma_ref_sets.size() >= this->n_ref_sets,ExcMessage("Called get_h_sigma with not enough datasets for the reference values of the independent fields!"));
	Assert(e_sigma_ref_sets[0].size() == this->e_sigma.size(),ExcMessage("Called get_h_sigma with invalid size of e_ref vector!"));

	const double D_x = e_sigma[0];
	const double D_y = e_sigma[1];
	const double D_z = e_sigma[2];
	const double phi = e_sigma[3];
	const double D_x_ref = e_sigma_ref_sets[0][0];
	const double D_y_ref = e_sigma_ref_sets[0][1];
	const double D_z_ref = e_sigma_ref_sets[0][2];
	const double phi_ref = e_sigma_ref_sets[0][3];

	const double n_x = n[0];
	const double n_y = n[1];
	const double n_z = spacedim == 3 ? n[2] : 0.0;

	//compute derivatives at alpha = 1.0
	if(get<0>(requested_quantities))
	{
		h_sigma = phi * (D_x*n_x + D_y*n_y + D_z*n_z);
	}

	if(get<1>(requested_quantities))
	{
		if(h_sigma_1.size() != this->e_sigma.size())
			h_sigma_1.reinit(this->e_sigma.size());
		h_sigma_1[0] = phi * n_x;
		h_sigma_1[1] = phi * n_y;
		h_sigma_1[2] = phi * n_z;
		h_sigma_1[3] = D_x*n_x + D_y*n_y + D_z*n_z;
	}
	if(get<2>(requested_quantities))
	{
		if( (h_sigma_2.size()[0] != this->e_sigma.size()) || (h_sigma_2.size()[1] != this->e_sigma.size()) )
			h_sigma_2.reinit(this->e_sigma.size(), this->e_sigma.size());
		h_sigma_2(0,3) = h_sigma_2(3,0) = n_x;
		h_sigma_2(1,3) = h_sigma_2(3,1) = n_y;
		h_sigma_2(2,3) = h_sigma_2(3,2) = n_z;
	}

	//now weight derivatives according to alpha
	if (alpha != 1.)
	{
		//gradient at reference state
		Vector<double>	h_sigma_1_ref;
		if(get<0>(requested_quantities) || get<1>(requested_quantities))
		{
			h_sigma_1_ref.reinit(e_sigma.size());
			h_sigma_1_ref[0] = phi_ref * n_x;
			h_sigma_1_ref[1] = phi_ref * n_y;
			h_sigma_1_ref[2] = phi_ref * n_z;
			h_sigma_1_ref[3] = D_x_ref*n_x + D_y_ref*n_y + D_z_ref*n_z;
		}

		if(get<0>(requested_quantities))
			h_sigma = h_sigma * alpha + h_sigma_1_ref * e_sigma * (1. - alpha);

		if(get<1>(requested_quantities))
			for(unsigned int m = 0; m < e_sigma.size(); ++m)
				h_sigma_1[m] = h_sigma_1[m] * alpha + h_sigma_1_ref[m] * (1. - alpha);

		if(get<2>(requested_quantities))
			h_sigma_2 *= alpha;
	}

	return false;
}

template class ElectroNeutralitySurface00<2>;
template class ElectroNeutralitySurface00<3>;
