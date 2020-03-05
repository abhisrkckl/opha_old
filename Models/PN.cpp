#include <cmath>
#include <ipow.hpp>

double PN_periastron_advance(double x, double e, double eta, double Xi){

    double q = (1 - sqrt(1-4*eta) - 2*eta)/(2.*eta);

    double k_3PN = (3*x)/(1 - e*e) + (ipow(x,2)*(54 + e*e*(51 - 26*eta) - 28*eta))/(4.*ipow(-1 + e*e,2)) + (ipow(x,3)*(e*e*(-96*(191 + 40*sqrt(1 - e*e)) + (-123*ipow(M_PI,2) + 64*(357 + 24*sqrt(1 - e*e) - 80*eta))*eta) + 4*(-240*(7 + 2*sqrt(1 - e*e)) + (-123*ipow(M_PI,2) + 8*(625 + 24*sqrt(1 - e*e) - 28*eta))*eta) - 16*ipow(e,4)*(156 + 5*eta*(-22 + 13*eta))))/(128.*ipow(-1 + e*e,3));
    double k_SO = -x*sqrt(x)*(4+3*q)*Xi/pow(1-e*e,1.5);

    return k_3PN + k_SO;
}

double PN_angular_eccentricity(double x, double e, double eta, double Xi){

    double q = 1-2*eta-sqrt(1-4*eta);

    double ephi = e*(1 + (4 - eta)*x + ((4*(-12*(26 + 15*sqrt(1 - ipow(e,2))) + eta*(17 + 72*sqrt(1 - ipow(e,2)) + eta)) + ipow(e,2)*(1152 + eta*(-659 + 41*eta)))*ipow(x,2))/(96.*(-1 + ipow(e,2))) + ((-70*ipow(e,4)*(-12288 + eta*(11233 + 5*eta*(-383 + 3*eta))) + 3*ipow(e,2)*(-8960*(76 + 35*sqrt(1 - ipow(e,2))) + eta*(4*(8*(12983 + 5040*sqrt(1 - ipow(e,2))) + 35*eta*(-1319 + 32*sqrt(1 - ipow(e,2)) + 45*eta)) - 1435*ipow(M_PI,2))) + 20*(1344*(54 + 65*sqrt(1 - ipow(e,2))) + eta*(4*(-33431 - 29960*sqrt(1 - ipow(e,2)) - 3458*eta + 3360*sqrt(1 - ipow(e,2))*eta) + 861*(1 + sqrt(1 - ipow(e,2)))*ipow(M_PI,2))))*ipow(x,3))/(26880.*ipow(-1 + ipow(e,2),2)));
    
    ephi += -e * x*sqrt(x)*(2+2*q)*Xi/pow(1-e*e,1.5);
    
    return ephi;
}

double PN_mean_motion(double x, double e, double eta, double Xi, double M){
    
        double nb = sqrt(x*x*x)/M;
        double k = PN_periastron_advance(x, e, eta, Xi);
        return nb / (1+k);    
    
}

double PN_coeff_Ft(double x, double e, double eta){
    return -(e*ipow(x,2)*(-15 + eta)*eta)/(8.*sqrt(1 - ipow(e,2))) - (ipow(x,3)*(-15 + eta)*eta*(5*ipow(e,2)*(3 - 5*eta) + 4*ipow(e,4)*(-15 + eta) - 3*(9 + eta)))/(192.*e*ipow(1 - ipow(e,2),1.5)) - (ipow(x,3)*(35*(-15 + eta)*eta*(9 + eta) - 35*ipow(e,4)*eta*(297 + eta*(-175 + 23*eta)) + ipow(e,2)*(-22400 + eta*(-28531 + 1435*ipow(M_PI,2) + 35*eta*(430 + 11*eta)))))/(2240.*e*ipow(1 - ipow(e,2),1.5));
}

double PN_coeff_Gt(double x, double e, double eta){
    return (-3*ipow(x,2)*(-5 + 2*eta))/(2.*sqrt(1 - ipow(e,2))) + (ipow(x,3)*(-5 + 2*eta)*(7*(9 + eta) + ipow(e,2)*(9 + 17*eta)))/(16.*ipow(1 - ipow(e,2),1.5)) + (ipow(x,3)*(6660 + 41*(-292 + 3*ipow(M_PI,2))*eta + 792*ipow(eta,2) + 36*ipow(e,2)*(95 + eta*(-55 + 18*eta))))/(192.*ipow(1 - ipow(e,2),1.5));
}

double PN_coeff_It(double x, double e, double eta){
    return (ipow(e,2)*ipow(x,3)*eta*(116 + eta*(-49 + 3*eta)))/(16.*ipow(1 - ipow(e,2),1.5));
}

double PN_coeff_Ht(double x, double e, double eta){
    return (ipow(e,3)*ipow(x,3)*eta*(23 + eta*(-73 + 13*eta)))/(192.*ipow(1 - ipow(e,2),1.5));
}

double PN_coeff_Fphi(double x, double e, double eta){
    return (ipow(e,2)*ipow(x,2)*(1 + (19 - 3*eta)*eta))/(8.*ipow(-1 + ipow(e,2),2)) - (ipow(x,3)*(-1 + eta*(-19 + 3*eta))*(3*(9 + eta) + ipow(e,4)*(-21 + 19*eta) + ipow(e,2)*(-6 + 26*eta)))/(96.*ipow(-1 + ipow(e,2),3)) + (ipow(x,3)*(840*(9 + eta)*(-1 + eta*(-19 + 3*eta)) - 280*ipow(e,4)*(3 + eta*(506 - 357*eta + 36*ipow(eta,2))) + ipow(e,2)*(-58800 + eta*(30135*ipow(M_PI,2) + 16*(-44284 + 105*eta*(136 + 7*eta))))))/(26880.*ipow(-1 + ipow(e,2),3));
}

double PN_coeff_Gphi(double x, double e, double eta){
    return (ipow(e,3)*ipow(x,2)*(1 - 3*eta)*eta)/(32.*ipow(-1 + ipow(e,2),2)) - (e*ipow(x,3)*eta*(-1 + 3*eta)*(9*(9 + eta) + 10*ipow(e,2)*(-9 + 7*eta) + ipow(e,4)*(9 + 17*eta)))/(768.*ipow(-1 + ipow(e,2),3)) + (e*ipow(x,3)*eta*(105*(9 + eta)*(-1 + 3*eta) + ipow(e,2)*(-34726 + 1435*ipow(M_PI,2) + 70*eta*(344 + eta)) - 35*ipow(e,4)*(14 + eta*(-49 + 26*eta))))/(8960.*ipow(-1 + ipow(e,2),3));
}

double PN_coeff_Iphi(double x, double e, double eta){
    return (ipow(e,4)*ipow(x,3)*eta*(-82 + 3*(19 - 5*eta)*eta))/(192.*ipow(-1 + ipow(e,2),3));
}

double PN_coeff_Hphi(double x, double e, double eta){
    return (ipow(e,5)*ipow(x,3)*eta*(-1 - 5*(-1 + eta)*eta))/(256.*ipow(-1 + ipow(e,2),3));
}


