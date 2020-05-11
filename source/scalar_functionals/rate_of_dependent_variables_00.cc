#include <deal.II/base/exceptions.h>
#include <deal.II/lac/vector.h>

#include <incremental_fe/scalar_functionals/rate_of_dependent_variables_00.h>

using namespace dealii;
using namespace std;
using namespace dealii::GalerkinTools;
using namespace incrementalFE;

template<unsigned int spacedim>
RateOfDependentVariables00<spacedim, spacedim>::RateOfDependentVariables00(	DependentField<spacedim,spacedim>	e_omega,
																			const set<types::material_id>		domain_of_integration,
																			const Quadrature<spacedim>			quadrature,
																		GlobalDataIncrementalFE<spacedim>&		global_data)
:
ScalarFunctional<spacedim, spacedim>({e_omega}, domain_of_integration, quadrature, "Rate of dependent variables", 1),
global_data(&global_data)
{
}

template<unsigned int spacedim>
bool
RateOfDependentVariables00<spacedim, spacedim>::get_h_omega(const Vector<double>&			e_omega,
															const vector<Vector<double>>&	e_omega_ref_sets,
															Vector<double>&					/*hidden_vars*/,
															const Point<spacedim>&			/*x*/,
															double&							h_omega,
															Vector<double>&					h_omega_1,
															FullMatrix<double>&				h_omega_2,
															const tuple<bool, bool, bool>	requested_quantities)
const
{

	Assert(e_omega.size() == this->e_omega.size(), ExcMessage("Called get_h_omega with invalid size of e vector!"));
	Assert(e_omega_ref_sets.size() >= this->n_ref_sets, ExcMessage("Called get_h_omega with not enough datasets for the reference values of the independent fields!"));
	Assert(e_omega_ref_sets[0].size() == this->e_omega.size(), ExcMessage("Called get_h_omega with invalid size of e_omega_ref_sets vector!"));

	const double delta_t = global_data->get_t()-global_data->get_t_ref();

	if(get<0>(requested_quantities) )
		h_omega = (e_omega[0] - e_omega_ref_sets[0][0])/delta_t;

	if(get<1>(requested_quantities))
	{
		if(h_omega_1.size() != this->e_omega.size())
			h_omega_1.reinit(this->e_omega.size());
		h_omega_1 = 1./delta_t;
	}

	if(get<2>(requested_quantities))
		if( (h_omega_2.size()[0]!=this->e_omega.size()) || (h_omega_2.size()[1]!=this->e_omega.size()) )
			h_omega_2.reinit(this->e_omega.size(), this->e_omega.size());

	return false;
}

template<unsigned int dim, unsigned int spacedim>
RateOfDependentVariables00<dim, spacedim>::RateOfDependentVariables00(	DependentField<dim,spacedim>		e_sigma,
																		const set<types::material_id>		domain_of_integration,
																		const Quadrature<dim>				quadrature,
																		GlobalDataIncrementalFE<spacedim>&	global_data):
ScalarFunctional<spacedim-1, spacedim>({e_sigma}, domain_of_integration, quadrature, "Rate of dependent variables", 1),
global_data(&global_data)
{
}

template<unsigned int dim, unsigned int spacedim>
bool
RateOfDependentVariables00<dim, spacedim>::get_h_sigma(	const Vector<double>& 			e_sigma,
														const vector<Vector<double>>&	e_sigma_ref,
														Vector<double>& 				/*hidden_vars*/,
														const Point<spacedim>& 			/*x*/,
														const Tensor<1,spacedim>& 		/*n*/,
														double& 						h_sigma,
														Vector<double>& 				h_sigma_1,
														FullMatrix<double>& 			h_sigma_2,
														const tuple<bool, bool, bool>	requested_quantities)
const
{
	Assert(e_sigma.size() == this->e_sigma.size(), ExcMessage("Called get_h_sigma with invalid size of e vector!"));
	Assert(e_sigma_ref.size() >= this->n_ref_sets, ExcMessage("Called get_h_sigma with not enough datasets for the reference values of the independent fields!"));
	Assert(e_sigma_ref[0].size() == this->e_sigma.size(), ExcMessage("Called get_h_sigma with invalid size of e_ref vector!"));

	const double delta_t = global_data->get_t()-global_data->get_t_ref();

	if(get<0>(requested_quantities) )
		h_sigma = (e_sigma[0] - e_sigma_ref[0][0])/delta_t;

	if(get<1>(requested_quantities))
	{
		if(h_sigma_1.size() != this->e_sigma.size())
			h_sigma_1.reinit(this->e_sigma.size());
		h_sigma_1 = 1.0/delta_t;
	}

	if(get<2>(requested_quantities))
		if( (h_sigma_2.size()[0]!=this->e_sigma.size()) || (h_sigma_2.size()[1]!=this->e_sigma.size()) )
			h_sigma_2.reinit(this->e_sigma.size(), this->e_sigma.size());

	return false;
}

template class RateOfDependentVariables00<2,2>;
template class RateOfDependentVariables00<3,3>;
template class RateOfDependentVariables00<1,2>;
template class RateOfDependentVariables00<2,3>;
