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

	vector<DependentField<spacedim, spacedim>> dependent_fields(9, DependentField<spacedim, spacedim>("q"));

	const double RT = 1.5;
	const double alpha = 1.0;
	const double n_0 = 0.1;
	const double V_m_f = 0.3;
	const double chi = 0.4;

	GlobalDataIncrementalFE<spacedim> global_data;
	global_data.set_t(1.0);

	PsiChemical06<spacedim> chemical_potential(	dependent_fields,
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
	e_omega[0] = 0.17400;
	e_omega[1] = 0.65967;
	e_omega[2] = 0.55345;
	e_omega[3] = 0.96550;
	e_omega[4] = 0.29811;
	e_omega[5] = 0.73196;
	e_omega[6] = 0.72997;
	e_omega[7] = 0.79393;
	e_omega[8] = 0.10799;

	e_omega_ref_sets[0] = e_omega;



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
