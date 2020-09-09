#include <iostream>
#include <math.h>

#include <deal.II/fe/fe_nothing.h>
#include <deal.II/base/quadrature_lib.h>

#include <incremental_fe/scalar_functionals/psi_lib.h>

using namespace std;
using namespace dealii;
using namespace dealii::GalerkinTools;
using namespace incrementalFE;

// Function defining scaling of Lame parameters within fluid domain dependent on position
template <unsigned int spacedim>
class LameScaling : public Function<spacedim>
{
	double
	value(	const Point<spacedim>&	/*p*/,
			const unsigned int		/*component=0*/)
	const
	{
		return 1.0;
	}
};

template<unsigned int spacedim>
void
check()
{

	vector<DependentField<spacedim, spacedim>> dependent_fields(9, DependentField<spacedim, spacedim>("q"));

	const double lambda = 1.5;
	const double mu = 1.3;
	const double alpha = 0.5;
	const double J_0 = 1.3;

	GlobalDataIncrementalFE<spacedim> global_data;
	global_data.set_t(1.0);

	LameScaling<spacedim> lame_scaling;
	PsiNeoHooke00<spacedim> psi_neo_hooke_00(	dependent_fields,
												{},
												QGauss<spacedim>(1),
												global_data,
												lambda,
												mu,
												lame_scaling,
												alpha,
												nullptr,
												J_0);

	Vector<double> e_omega(dependent_fields.size());
	vector<Vector<double>> e_omega_ref_sets(1);
	e_omega_ref_sets[0].reinit(dependent_fields.size());
	Vector<double> hidden_vars;
	Point<spacedim> x;

	double a = 0.1;
	for(auto& val : e_omega)
	{
		val = cos(a);
		a = a + 0.1;
	}
	e_omega[0] += 2.0;
	e_omega[4] += 2.0;
	e_omega[8] += 2.0;

	for(auto& val : e_omega_ref_sets[0])
	{
		val = sin(a);
		a = a + 0.1;
	}
	e_omega_ref_sets[0][0] += 2.0;
	e_omega_ref_sets[0][4] += 2.0;
	e_omega_ref_sets[0][8] += 2.0;

	psi_neo_hooke_00.compare_derivatives_with_numerical_derivatives(e_omega,
																	e_omega_ref_sets,
																	hidden_vars,
																	x,
																	"derivatives.dat");
}


int main()
{
	cout << "### 2D-Case ###\n\n";
	check<2>();

	cout << "\n### 3D-Case ###\n\n";
	check<3>();
}
