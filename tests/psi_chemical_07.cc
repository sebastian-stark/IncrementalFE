#include <iostream>
#include <math.h>

#include <deal.II/fe/fe_nothing.h>
#include <deal.II/base/quadrature_lib.h>

#include <incremental_fe/scalar_functionals/psi_lib.h>

using namespace std;
using namespace dealii;
using namespace dealii::GalerkinTools;
using namespace incrementalFE;

template<unsigned int spacedim>
void
check()
{

	vector<DependentField<spacedim, spacedim>> dependent_fields(1, DependentField<spacedim, spacedim>("q"));

	const double RT = 1.5;
	const double alpha = 1.0;
	const double n_0 = 0.1;
	const double V_m_f = 0.3;
	const double chi = 0.4;

	GlobalDataIncrementalFE<spacedim> global_data;
	global_data.set_t(1.0);

	PsiChemical07<spacedim> chemical_potential(	dependent_fields,
												{},
												QGauss<spacedim>(1),
												global_data,
												RT,
												n_0,
												V_m_f,
												chi,
												alpha);

	Vector<double> e_omega(dependent_fields.size());
	vector<Vector<double>> e_omega_ref_sets(1);
	e_omega_ref_sets[0].reinit(dependent_fields.size());
	Vector<double> hidden_vars;
	Point<spacedim> x;
	e_omega[0] = 0.174;
	e_omega_ref_sets[0] = 0.36;

	chemical_potential.compare_derivatives_with_numerical_derivatives(	e_omega,
																		e_omega_ref_sets,
																		hidden_vars,
																		x);
}


int main()
{
	cout << "### 2D-Case ###\n\n";
	check<2>();

	cout << "\n### 3D-Case ###\n\n";
	check<3>();
}
