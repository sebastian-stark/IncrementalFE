#include <incremental_fe/scalar_functionals/pressure_work_00.h>

#include <math.h>

#include <deal.II/base/exceptions.h>
#include <deal.II/lac/vector.h>

using namespace std;
using namespace dealii;
using namespace dealii::GalerkinTools;
using namespace incrementalFE;

template<unsigned int spacedim>
void
PressureWork00<spacedim>::compute_derivatives(	const Vector<double>&			e_sigma,
												const Tensor<1,spacedim>& 		n,
												double&							val,
												Vector<double>&					d1,
												FullMatrix<double>& 			d2,
												const tuple<bool, bool, bool>&	requested_quantities)
const
{
	//deformation gradient
	Tensor<2, 3> F;
	for(unsigned int m = 0; m < 3; ++m)
		for(unsigned int k = 0; k < 3; ++k)
			F[m][k] = e_sigma[4 + 3 * m + k];

	Tensor<1, 3> n_, F_n;
	for(unsigned int m = 0; m < spacedim; ++m)
		n_[m] = n[m];
	F_n = F * n_;

	const double p = e_sigma[0];

	val = 0.0;

	//first derivative
	if(get<1>(requested_quantities))
	{
		d1.reinit(e_sigma.size());
		for(unsigned int m = 0; m < 3; ++m)
			d1[m + 1] = p * F_n[m];
	}

	//second derivative
	if(get<2>(requested_quantities))
	{
		d2.reinit(e_sigma.size(),e_sigma.size());
		for(unsigned int m = 0; m < 3; ++m)
			d2[m + 1][0] = F_n[m];


		for(unsigned int m = 0; m < 3; ++m)
			for(unsigned int M = 0; M < 3; ++M)
				d2[m + 1][ 4 + 3 * m + M] = p * n_[M];
	}
}

template<unsigned int spacedim>
PressureWork00<spacedim>::PressureWork00(	const vector<DependentField<spacedim-1,spacedim>>	e_sigma,
											const set<types::material_id>						domain_of_integration,
											const Quadrature<spacedim-1>						quadrature,
											GlobalDataIncrementalFE<spacedim>&					global_data,
											const double										alpha)
:
ScalarFunctional<spacedim-1, spacedim>(e_sigma, domain_of_integration, quadrature, "PressureWork00", 1),
global_data(&global_data),
alpha(alpha)
{
	Assert( (alpha >= 0.0) && (alpha <= 1.0), ExcMessage("alpha must be in the range 0 <= alpha <= 1 !"));
}

template<unsigned int spacedim>
bool
PressureWork00<spacedim>::get_h_sigma(	const Vector<double>&			e_sigma,
										const vector<Vector<double>>&	e_sigma_ref_sets,
										Vector<double>&					/*hidden_vars*/,
										const Point<spacedim>&			/*x*/,
										const Tensor<1,spacedim>& 		n,
										double&							h_sigma,
										Vector<double>&					h_sigma_1,
										FullMatrix<double>&				h_sigma_2,
										const tuple<bool, bool, bool>	requested_quantities)
const
{

	Assert(e_sigma.size() == this->e_sigma.size(),ExcMessage("Called get_h_sigma with invalid size of e vector!"));
	Assert(e_sigma_ref_sets.size() >= this->n_ref_sets,ExcMessage("Called get_h_sigma with not enough datasets for the reference values of the independent fields!"));
	Assert(e_sigma_ref_sets[0].size() == this->e_sigma.size(),ExcMessage("Called get_h_sigma with invalid size of e_sigma_ref vector!"));

	//compute derivatives at alpha = 1.0
	compute_derivatives(e_sigma, n, h_sigma, h_sigma_1, h_sigma_2, requested_quantities);

	//now weight derivatives according to alpha
	if (alpha != 1.)
	{
		//gradient at reference state
		Vector<double>	h_sigma_1_ref;
		if(get<0>(requested_quantities) || get<1>(requested_quantities))
		{
			h_sigma_1_ref.reinit(e_sigma.size());
			compute_derivatives(e_sigma_ref_sets[0], n, h_sigma, h_sigma_1_ref, h_sigma_2, make_tuple(false, true, false));
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

template class PressureWork00<2>;
template class PressureWork00<3>;
