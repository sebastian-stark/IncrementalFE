#include <deal.II/base/exceptions.h>
#include <deal.II/lac/vector.h>

#include <incremental_fe/scalar_functionals/linear_material_00.h>

using namespace dealii;
using namespace std;
using namespace dealii::GalerkinTools;
using namespace incrementalFE;

template<unsigned int spacedim>
LinearMaterial00<spacedim, spacedim>::LinearMaterial00(	const vector<DependentField<spacedim,spacedim>>	e_omega,
														const set<types::material_id>					domain_of_integration,
														const Quadrature<spacedim>						quadrature,
														GlobalDataIncrementalFE<spacedim>&				global_data,
														const FullMatrix<double>						C,
														const Vector<double>							y,
														const string									name,
														const double									alpha,
														const unsigned int								method)
:
ScalarFunctional<spacedim, spacedim>(e_omega, domain_of_integration, quadrature, name, 1),
global_data(&global_data),
C(C),
y(y),
alpha(alpha),
method(method)
{
	Assert(	(C.size()[0] == e_omega.size()) && (C.size()[1] == e_omega.size()),
			ExcMessage("Matrix of linear material must be square and have same the dimension as the number of dependent fields!") );
	Assert(	(y.size() == e_omega.size()),
			ExcMessage("Vector of linear material must have same dimension as number of dependent fields!") );
	Assert( (alpha >= 0.0) && (alpha <= 1.0),
			ExcMessage("alpha must be in the range 0 <= alpha <= 1 !"));
}

template<unsigned int spacedim>
bool
LinearMaterial00<spacedim, spacedim>::get_h_omega(	const Vector<double>&			e_omega,
													const vector<Vector<double>>&	e_omega_ref_sets,
													Vector<double>&					/*hidden_vars*/,
													const Point<spacedim>&			/*x*/,
													double&							h_omega,
													Vector<double>&					h_omega_1,
													FullMatrix<double>&				h_omega_2,
													const tuple<bool, bool, bool>	requested_quantities)
const
{

	Assert(e_omega.size()==this->e_omega.size(),ExcMessage("Called get_h_omega with invalid size of e vector!"));
	Assert(e_omega_ref_sets.size()>=this->n_ref_sets,ExcMessage("Called get_h_omega with not enough datasets for the reference values of the independent fields!"));
	Assert(e_omega_ref_sets[0].size()==this->e_omega.size(),ExcMessage("Called get_h_omega with invalid size of e_omega_ref_sets vector!"));

	if(method == 0)
	{

		Vector<double> e_omega_alpha(e_omega.size());
		for(unsigned int m = 0; m < e_omega.size(); ++m)
			e_omega_alpha[m] = alpha*e_omega[m] + (1. - alpha)*e_omega_ref_sets[0][m];

		Vector<double> Ce(this->e_omega.size());
			C.vmult(Ce, e_omega_alpha);

		if(get<0>(requested_quantities) )
		{
			h_omega = 0.5*(e_omega_alpha*Ce);
			h_omega += e_omega_alpha*y;
			h_omega = h_omega/alpha;
		}

		if(get<1>(requested_quantities))
		{
			if(h_omega_1.size() != this->e_omega.size())
				h_omega_1.reinit(this->e_omega.size());
			h_omega_1 = y;
			h_omega_1 += Ce;
		}
		if(get<2>(requested_quantities))
		{
			if( (h_omega_2.size()[0] != this->e_omega.size()) || (h_omega_2.size()[1] != this->e_omega.size()) )
				h_omega_2.reinit(this->e_omega.size(), this->e_omega.size());
			h_omega_2 = C;
			h_omega_2 *= alpha;
		}
	}
	else
	{
		Vector<double> Ce(this->e_omega.size());
			C.vmult(Ce, e_omega);

		if(get<0>(requested_quantities) )
		{
			h_omega = 0.5*(e_omega*Ce);
			h_omega += e_omega*y;
		}

		if(get<1>(requested_quantities))
		{
			if(h_omega_1.size() != this->e_omega.size())
				h_omega_1.reinit(this->e_omega.size());
			h_omega_1 = y;
			h_omega_1 += Ce;
		}

		if(get<2>(requested_quantities))
		{
			if( (h_omega_2.size()[0] != this->e_omega.size()) || (h_omega_2.size()[1] != this->e_omega.size()) )
				h_omega_2.reinit(this->e_omega.size(), this->e_omega.size());
			h_omega_2 = C;
		}

		if(alpha != 1.)
		{
			//gradient at reference state
			Vector<double>	h_omega_1_ref;
			if(get<0>(requested_quantities) || get<1>(requested_quantities))
			{
				h_omega_1_ref.reinit(e_omega.size());
				h_omega_1_ref = y;
				C.vmult(h_omega_1_ref, e_omega_ref_sets[0], true);
			}

			if(get<0>(requested_quantities))
				h_omega = h_omega * alpha + h_omega_1_ref*e_omega *(1. - alpha);

			if(get<1>(requested_quantities))
				for(unsigned int m=0; m < e_omega.size(); ++m)
					h_omega_1[m] = h_omega_1[m] * alpha + h_omega_1_ref[m] * (1. - alpha);

			if(get<2>(requested_quantities))
				h_omega_2 *= alpha;
		}
	}

	return false;
}

template<unsigned int dim, unsigned int spacedim>
LinearMaterial00<dim,spacedim>::LinearMaterial00(	const vector<DependentField<dim,spacedim>>	e_sigma,
													const set<types::material_id>				domain_of_integration,
													const Quadrature<dim>						quadrature,
													GlobalDataIncrementalFE<spacedim>&			global_data,
													const FullMatrix<double>					C,
													const Vector<double>						y,
													const string								name,
													const double								alpha,
													const unsigned int							method)
:
ScalarFunctional<spacedim-1, spacedim>(e_sigma, domain_of_integration, quadrature, name, 1),
global_data(&global_data),
C(C),
y(y),
alpha(alpha),
method(method)
{
	Assert(	(C.size()[0] == e_sigma.size()) && (C.size()[1] == e_sigma.size()),
			ExcMessage("Matrix of linear material must be square and have same the dimension as the number of dependent fields!") );
	Assert(	(y.size() == e_sigma.size()),
			ExcMessage("Vector of linear material must have same dimension as number of dependent fields!") );
	Assert( (alpha >= 0.0) && (alpha <= 1.0),
			ExcMessage("alpha must be in the range 0 <= alpha <= 1 !"));
}

template<unsigned int dim, unsigned int spacedim>
bool
LinearMaterial00<dim, spacedim>::get_h_sigma(	const Vector<double>& 			e_sigma,
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
	Assert(e_sigma.size() == this->e_sigma.size(),ExcMessage("Called get_h_sigma with invalid size of e vector!"));
	Assert(e_sigma_ref.size() >= this->n_ref_sets,ExcMessage("Called get_h_sigma with not enough datasets for the reference values of the independent fields!"));
	Assert(e_sigma_ref[0].size() == this->e_sigma.size(),ExcMessage("Called get_h_sigma with invalid size of e_ref vector!"));

	if(method == 0)
	{
		Vector<double> e_sigma_(e_sigma.size());
		for(unsigned int m=0; m<e_sigma.size(); ++m)
			e_sigma_[m] = alpha*e_sigma[m] + (1.-alpha)*e_sigma_ref[0][m];

		Vector<double> Ce_sigma(this->e_sigma.size());
			C.vmult(Ce_sigma, e_sigma_);

		if(get<0>(requested_quantities))
		{
			h_sigma = 0.5*(e_sigma_*Ce_sigma);
			h_sigma += e_sigma_*y;
			h_sigma = h_sigma/alpha;
		}

		if(get<1>(requested_quantities))
		{
			if(h_sigma_1.size() != this->e_sigma.size())
				h_sigma_1.reinit(this->e_sigma.size());
			h_sigma_1 = y;
			h_sigma_1 += Ce_sigma;
		}

		if(get<2>(requested_quantities))
		{
			if( (h_sigma_2.size()[0] != this->e_sigma.size()) || (h_sigma_2.size()[1] != this->e_sigma.size()) )
				h_sigma_2.reinit(this->e_sigma.size(), this->e_sigma.size());
			h_sigma_2 = C;
			h_sigma_2 *= alpha;
		}
	}
	else
	{
		Vector<double> Ce_sigma(this->e_sigma.size());
			C.vmult(Ce_sigma,e_sigma);

		if(get<0>(requested_quantities))
		{
			h_sigma = 0.5*(e_sigma*Ce_sigma);
			h_sigma += e_sigma*y;
		}

		if(get<1>(requested_quantities))
		{
			if(h_sigma_1.size() != this->e_sigma.size())
				h_sigma_1.reinit(this->e_sigma.size());
			h_sigma_1 = y;
			h_sigma_1 += Ce_sigma;
		}
		if(get<2>(requested_quantities))
		{
			if( (h_sigma_2.size()[0] != this->e_sigma.size()) || (h_sigma_2.size()[1] != this->e_sigma.size()) )
				h_sigma_2.reinit(this->e_sigma.size(), this->e_sigma.size());
			h_sigma_2 = C;
		}

		if(alpha != 1.)
		{
			//gradient at reference state
			Vector<double>	h_sigma_1_ref;
			if(get<0>(requested_quantities) || get<1>(requested_quantities))
			{
				h_sigma_1_ref.reinit(e_sigma.size());
				h_sigma_1_ref = y;
				C.vmult(h_sigma_1_ref, e_sigma_ref[0], true);
			}

			if(get<0>(requested_quantities))
				h_sigma = h_sigma * alpha + h_sigma_1_ref*e_sigma *(1. - alpha);

			if(get<1>(requested_quantities))
				for(unsigned int m = 0; m < e_sigma.size(); ++m)
					h_sigma_1[m] = h_sigma_1[m] * alpha + h_sigma_1_ref[m] * (1. - alpha);

			if(get<2>(requested_quantities))
				h_sigma_2 *= alpha;
		}
	}

	return false;
}

template class LinearMaterial00<2,2>;
template class LinearMaterial00<3,3>;
template class LinearMaterial00<1,2>;
template class LinearMaterial00<2,3>;
