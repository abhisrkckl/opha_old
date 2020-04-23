#include <cmath>
#include "ipow.hpp"
#include "Opha.hpp"
#include "python.hpp"

typedef Opha::Model<4,0,3,2> Spin;

// DO NOT TOUCH THIS FUNCTION.
// For a new model write your own.
template<>
void Spin::ODE_system::operator()(const state_t &state, state_t &derivatives_out, const double /*phi*/) const{
        
    const auto& [x,e,u,t] = state;
    const auto& [M,eta,Xi]   = params.bin_params();
    
    const double OTS  = sqrt(1-e*e),
                 DT   = 1-e*cos(u),
                 nb   = x*sqrt(x)/M,
                 q    = (1-2*eta-sqrt(1-4*eta))/(2*eta);
    
    // [ dphi/dt
    const double dphi_dt_N = nb*OTS/ipow(DT,2);
    const double dphi_dt_corr = ( 1 + ((-1 + DT + ipow(e,2))*(-4 + eta)*x)/(DT*ipow(OTS,2)) + ((DT*(108 + 63*eta + 33*ipow(eta,2))*ipow(OTS,4) - 6*eta*(3 + 2*eta)*ipow(OTS,6) + ipow(DT,2)*ipow(OTS,2)*(-240 - 31*eta - 29*ipow(eta,2) + ipow(e,2)*(48 - 17*eta + 17*ipow(eta,2)) + (180 - 72*eta)*OTS) + ipow(DT,3)*(42 + 22*eta + 8*ipow(eta,2) + ipow(e,2)*(-147 + 8*eta - 14*ipow(eta,2)) + (-90 + 36*eta)*OTS))*ipow(x,2))/(12.*ipow(DT,3)*ipow(OTS,4)) + ((1120*DT*eta*(-349 - 186*eta + 6*ipow(eta,2))*ipow(OTS,8) + 5040*eta*(-3 + 8*eta + 2*ipow(eta,2))*ipow(OTS,10) + 140*ipow(DT,3)*ipow(OTS,4)*(-4032 - 15688*eta + 1020*ipow(eta,2) + 724*ipow(eta,3) + ipow(e,2)*(1728 + 3304*eta - 612*ipow(eta,2) - 460*ipow(eta,3)) + (8640 - 5616*eta + 864*ipow(eta,2))*OTS) + 4*ipow(DT,2)*eta*ipow(OTS,6)*(539788 + 20160*eta - 19600*ipow(eta,2) + ipow(e,2)*(4200 - 5040*eta + 1120*ipow(eta,2)) - 4305*ipow(M_PI,2)) + 4*ipow(DT,4)*ipow(OTS,2)*(127680 - 32900*ipow(eta,2) - 11060*ipow(eta,3) + ipow(e,4)*(4620*eta + 3220*ipow(eta,2) - 4060*ipow(eta,3)) + eta*(19372 + 12915*ipow(M_PI,2)) + ipow(e,2)*(-252000 + 98560*ipow(eta,2) + 16800*ipow(eta,3) + (134400 - 119280*eta + 40320*ipow(eta,2))*OTS + eta*(-300528 + 4305*ipow(M_PI,2))) + OTS*(-235200 + eta*(-162400 + 4305*ipow(M_PI,2)))) + ipow(DT,5)*(-147840 + 8960*ipow(eta,2) + 4480*ipow(eta,3) + ipow(e,4)*(-221760 - 113680*eta + 94640*ipow(eta,2) + 13440*ipow(eta,3)) + eta*(1127280 - 43050*ipow(M_PI,2)) + OTS*(-67200 - 53760*ipow(eta,2) + eta*(674240 - 8610*ipow(M_PI,2))) + ipow(e,2)*(-194880 - 112000*ipow(eta,2) - 11200*ipow(eta,3) + (-739200 + 544320*eta - 127680*ipow(eta,2))*OTS + eta*(692928 + 12915*ipow(M_PI,2)))))*ipow(x,3))/(13440.*ipow(DT,5)*ipow(OTS,6)));
    const double dphi_dt_SO = e*(cos(u)-e)/ipow(OTS,3)/DT*sqrt(x*x*x)*(2+2*q)*Xi;
    const double dphi_dt = dphi_dt_N*(dphi_dt_corr+dphi_dt_SO);
    // ]
    
    // [ dx/dt
    const double dx_dt_conservative = (eta*ipow(x,5)*(12.8 + (584*ipow(e,2))/15. + (74*ipow(e,4))/15. + ((-11888 + ipow(e,2)*(87720 - 159600*eta) + ipow(e,4)*(171038 - 141708*eta) + ipow(e,6)*(11717 - 8288*eta) - 14784*eta)*x)/(420.*ipow(OTS,2)) + ((-360224 + 4514976*eta + 1903104*ipow(eta,2) + ipow(e,8)*(3523113 - 3259980*eta + 1964256*ipow(eta,2)) + ipow(e,2)*(-92846560 + 15464736*eta + 61282032*ipow(eta,2)) + ipow(e,6)*(83424402 - 123108426*eta + 64828848*ipow(eta,2)) + ipow(e,4)*(783768 - 207204264*eta + 166506060*ipow(eta,2)) - 3024*(96 + 4268*ipow(e,2) + 4386*ipow(e,4) + 175*ipow(e,6))*(-5 + 2*eta)*OTS)*ipow(x,2))/(45360.*ipow(OTS,4))))/(M*ipow(OTS,7));
    const double tail_func_x_Pade = (1 + (3436949070776656074932189.*ipow(e,2))/4.7767337158899716e23 + (184431544784455079079876287.*ipow(e,4))/3.4392482754407795e25 + (835839702422394923892514501.*ipow(e,6))/1.9652847288233026e27)/(ipow(1 - ipow(e,2),5)*(1 + (6041671199700943388399.*ipow(e,2))/1.7912751434587393e23 - (7703779407266542976249.*ipow(e,4))/2.866040229533983e24 + (80463025194022585753159.*ipow(e,6))/2.866040229533983e26 - (1311480865588092344321.*ipow(e,8))/7.503814419143519e25));
    const double dx_dt_tail    = 256*eta*M_PI * sqrt(ipow(x,13)) / (5.*M)  * tail_func_x_Pade;
    const double dx_dt = dx_dt_conservative + dx_dt_tail;
    // ]

    // [ de/dt
    const double de_dt_conservative = -((e*eta*ipow(x,4)*((304 + 121*ipow(e,2))/(15.*ipow(OTS,5)) - ((67608 + 228704*eta + ipow(e,4)*(-125361 + 93184*eta) + ipow(e,2)*(-718008 + 651252*eta))*x)/(2520.*ipow(OTS,7)) + ((-502.5804232804233 + (18763*eta)/42. + (752*ipow(eta,2))/5. + ipow(e,6)*(125.21636904761905 - (362071*eta)/2520. + (821*ipow(eta,2))/9.) + ipow(e,4)*(1540.3345899470899 - (13018711*eta)/5040. + (127411*ipow(eta,2))/90.) + ipow(e,2)*(-1223.3265873015873 - (988423*eta)/840. + (64433*ipow(eta,2))/40.) + (445.3333333333333 + ipow(e,2)*(1160.5 - (2321*eta)/5.) + ipow(e,4)*(94.16666666666667 - (113*eta)/3.) - (2672*eta)/15.)*OTS)*ipow(x,2))/ipow(OTS,9)))/M);
    const double tail_func_e_Pade = (192*sqrt(1 - ipow(e,2))*((1 + (3436949070776656074932189.*ipow(e,2))/4.7767337158899716e23 + (184431544784455079079876287.*ipow(e,4))/3.4392482754407795e25 + (835839702422394923892514501.*ipow(e,6))/1.9652847288233026e27)/(ipow(1 - ipow(e,2),4.5)*(1 + (6041671199700943388399.*ipow(e,2))/1.7912751434587393e23 - (7703779407266542976249.*ipow(e,4))/2.866040229533983e24 + (80463025194022585753159.*ipow(e,6))/2.866040229533983e26 - (1311480865588092344321.*ipow(e,8))/7.503814419143519e25)) - (1. + 3.021247735*ipow(e,2) + 0.3546477902*ipow(e,4))/(ipow(1. - 1.*ipow(e,2),3.5)*(1. - 0.5605225135999999*e - ipow(e,2) - 2*ipow(e,4) - 4*ipow(e,6) - 4*ipow(e,8) - 5*ipow(e,10)))))/(985.*ipow(e,2));
    const double de_dt_tail = (-394*e*eta*M_PI*sqrt(ipow(x,11)))/(3.*M) * tail_func_e_Pade;
    const double de_dt = de_dt_conservative + de_dt_tail;
    // ]
    
    // [ du/dt
    const double du_dt_N = nb/DT;
    const double du_dt_corr = 1 - (3*x)/ipow(OTS,2) + ((-15*eta + ipow(eta,2) + ipow(e,2)*(45*eta - 3*ipow(eta,2)) + ipow(e,6)*(15*eta - ipow(eta,2)) + ipow(e,4)*(-45*eta + 3*ipow(eta,2)) + ipow(DT,3)*(-36 + 56*eta + ipow(e,2)*(-102 + 52*eta)) + DT*(-60 + 39*eta - ipow(eta,2) + ipow(e,4)*(-60 + 39*eta - ipow(eta,2)) + ipow(e,2)*(120 - 78*eta + 2*ipow(eta,2))) + ipow(DT,2)*(60 - 24*eta + ipow(e,2)*(-60 + 24*eta))*OTS)*ipow(x,2))/(8.*ipow(DT,3)*ipow(OTS,4)) + ((ipow(DT,6)*(-201600 + ipow(e,4)*(403200 - 161280*eta) + 80640*eta + ipow(e,2)*(-201600 + 80640*eta)) + ipow(DT,5)*(-100800 - 640640*eta + 67200*ipow(eta,2) + ipow(e,6)*(201600 - 194880*eta + 73920*ipow(eta,2)) + 8610*eta*ipow(M_PI,2) + ipow(e,2)*(403200 + 1086400*eta - 60480*ipow(eta,2) - 17220*eta*ipow(M_PI,2)) + ipow(e,4)*(-504000 - 250880*eta - 80640*ipow(eta,2) + 8610*eta*ipow(M_PI,2))) + OTS*(ipow(DT,2)*(-735000*eta + 108360*ipow(eta,2) + 15960*ipow(eta,3) + ipow(e,2)*(2940000*eta - 433440*ipow(eta,2) - 63840*ipow(eta,3)) + ipow(e,6)*(2940000*eta - 433440*ipow(eta,2) - 63840*ipow(eta,3)) + ipow(e,8)*(-735000*eta + 108360*ipow(eta,2) + 15960*ipow(eta,3)) + ipow(e,4)*(-4410000*eta + 650160*ipow(eta,2) + 95760*ipow(eta,3))) + DT*(-19320*eta + 61320*ipow(eta,2) - 10920*ipow(eta,3) + ipow(e,4)*(-193200*eta + 613200*ipow(eta,2) - 109200*ipow(eta,3)) + ipow(e,8)*(-96600*eta + 306600*ipow(eta,2) - 54600*ipow(eta,3)) + ipow(e,10)*(19320*eta - 61320*ipow(eta,2) + 10920*ipow(eta,3)) + ipow(e,2)*(96600*eta - 306600*ipow(eta,2) + 54600*ipow(eta,3)) + ipow(e,6)*(193200*eta - 613200*ipow(eta,2) + 109200*ipow(eta,3))) + ipow(DT,6)*(20160 + 1535520*eta - 94080*ipow(eta,2) + ipow(e,4)*(-262080 + 184800*eta - 109200*ipow(eta,2)) - 51660*eta*ipow(M_PI,2) + ipow(e,2)*(-897120 + 1874880*eta - 537600*ipow(eta,2) - 12915*eta*ipow(M_PI,2))) + ipow(DT,3)*(-940800 + 1940784*eta - 368760*ipow(eta,2) + 1400*ipow(eta,3) + ipow(e,8)*(-15120*eta + 17640*ipow(eta,2) - 3640*ipow(eta,3)) + 8610*eta*ipow(M_PI,2) + ipow(e,2)*(2822400 - 5807232*eta + 1088640*ipow(eta,2) - 560*ipow(eta,3) - 25830*eta*ipow(M_PI,2)) + ipow(e,6)*(940800 - 1895424*eta + 315840*ipow(eta,2) + 9520*ipow(eta,3) - 8610*eta*ipow(M_PI,2)) + ipow(e,4)*(-2822400 + 5776992*eta - 1053360*ipow(eta,2) - 6720*ipow(eta,3) + 25830*eta*ipow(M_PI,2))) + ipow(DT,4)*(1041600 - 545824*eta + 131880*ipow(eta,2) - 6440*ipow(eta,3) + ipow(e,6)*(-201600 + 576240*eta - 202440*ipow(eta,2) + 4760*ipow(eta,3)) - 17220*eta*ipow(M_PI,2) + ipow(e,4)*(1444800 - 1698304*eta + 536760*ipow(eta,2) - 15960*ipow(eta,3) - 17220*eta*ipow(M_PI,2)) + ipow(e,2)*(-2284800 + 1667888*eta - 466200*ipow(eta,2) + 17640*ipow(eta,3) + 34440*eta*ipow(M_PI,2)))))*ipow(x,3))/(13440.*ipow(DT,6)*ipow(OTS,7));
    const double du_dt_SO = sqrt(x*x*x)*(4+3*q)*Xi/ipow(OTS,3);
    const double du_dt = du_dt_N*(du_dt_corr+du_dt_SO);
    // ]
    
    // [ Output
    derivatives_out[0] = dx_dt/dphi_dt;
    derivatives_out[1] = de_dt/dphi_dt;
    derivatives_out[2] = du_dt/dphi_dt;
    derivatives_out[3] =    1./dphi_dt;        
    // ]
}

template<>
double Spin::emission_delay(const params_t& params, const state_t& impact_state, const double phi){
    

	const auto& [x,e,u,t] = impact_state;
    const auto& [M,eta,Xi]   = params.bin_params();
    const auto& [de,dd]   = params.delay_params();

    constexpr double lts_to_AU = 0.0020039888;
    const double ar = M/x * (1-x/3*(9-eta)),
                 er = e   * (1+x/2*(883*eta)),
                 OTS = sqrt(1 - e*e),
                 d_t = (1-e*cos(u)),
                 r = ar*d_t,
                 phidot = pow(x,1.5) * OTS/ (M*d_t*d_t) * (1 + x*(-4+eta)*(-1+d_t + e*e)/(OTS*OTS*d_t)),
                 v_sec = r*phidot,
                 r_isco = 6*M,
                 r_AU = r*lts_to_AU;
                 //r_log = log10(r_AU);

    double delay_yr = pow(v_sec, -4.226) * pow((r/r_isco),-0.546) * pow((1 - (1/sqrt(r/r_isco))), 0.255);

    double delay_s = de*0.0001*delay_yr*365.25*24*3600;

    double r1 = r_AU/2635;
    double delay_d = (r1>1.7) ?(-dd/v_sec * sqrt((r1-1.7)/6) * sin((r1-4)/1.2)) : 0;
    
    //if(int(round(phi/M_PI))%2 == 1){
    //    double term5 = -0.00817*(r1-2400./527)*(r1-2400./527);
    //    delay_d += dc/v_sec * 1e10 * pow(10,term5);
    //}

    delay_d *= 365.25*24*3600;

    return delay_s + delay_d;
}

template<>
std::string Spin::description(){
    return "Post-Newtonian model (3PN conservative, 3.5PN reactive, 4PN tail, Spin-Orbit) with emission delay and disk deformation delay.\n  The parameters are [ x,e,u,t |  | M,eta,Xi | de,dd ].";
}

template<>
double Spin::radius(const params_t& params, const state_t& state, const double phi){
    
    const auto& [x,e,u,t] = state;
    const auto& [M,eta,Xi] = params.bin_params();
    
    const double q  = (1-2*eta-sqrt(1-4*eta))/(2*eta);
    
    const double ar = M/x*(1 + ((3 + ipow(e,2)*(-9 + eta) - eta)*x)/(3.*(-1 + ipow(e,2))) + ((-180*(-1 + sqrt(1 - ipow(e,2))) + eta*(99 + 72*sqrt(1 - ipow(e,2)) + 4*eta) + ipow(e,4)*(36 + eta*(15 + 4*eta)) - 2*ipow(e,2)*(-9*(21 + 10*sqrt(1 - ipow(e,2))) + eta*(219 + 36*sqrt(1 - ipow(e,2)) + 4*eta)))*ipow(x,2))/(36.*ipow(-1 + ipow(e,2),2)) - ((-201600*(-9 + 9*sqrt(1 - ipow(e,2)) + 34*eta) + 280*ipow(e,6)*(432*(15 + sqrt(1 - ipow(e,2))) + eta*(-27*(232 + 3*sqrt(1 - ipow(e,2))) + 2*eta*(1188 + 45*sqrt(1 - ipow(e,2)) + 8*sqrt(1 - ipow(e,2))*eta))) + eta*(216*(39819*sqrt(1 - ipow(e,2)) + 2800*eta) - 35*(16*sqrt(1 - ipow(e,2))*eta*(81 + 8*eta) + 1107*(-2 + 3*sqrt(1 - ipow(e,2)))*ipow(M_PI,2))) + 105*ipow(e,2)*(-1728*(40 + 89*sqrt(1 - ipow(e,2))) + eta*(24*(5608 + 7629*sqrt(1 - ipow(e,2)) - 216*eta) + 16*sqrt(1 - ipow(e,2))*eta*(-1335 + 8*eta) - 369*(4 + 9*sqrt(1 - ipow(e,2)))*ipow(M_PI,2))) - 3*ipow(e,4)*(60480*(-20 + 29*sqrt(1 - ipow(e,2))) + eta*(24*(76720 - 176643*sqrt(1 - ipow(e,2)) + 10080*eta) + 35*(32*sqrt(1 - ipow(e,2))*eta*(1311 + 4*eta) + 369*(-2 + 3*sqrt(1 - ipow(e,2)))*ipow(M_PI,2)))))*ipow(x,3))/(181440.*ipow(1 - ipow(e,2),3.5)) + ((2+q)*x*sqrt(x)*Xi)/sqrt(1 - ipow(e,2)));
    const double er = e*(1 + (4 - (3*eta)/2.)*x + ((-12*(16 + 15*sqrt(1 - ipow(e,2))) + (323 + 72*sqrt(1 - ipow(e,2)) - 21*eta)*eta + ipow(e,2)*(288 + eta*(-227 + 21*eta)))*ipow(x,2))/(24.*(-1 + ipow(e,2))) + ((6720*(4 + 65*sqrt(1 - ipow(e,2))) - 475388*eta - 140*ipow(e,4)*(-1536 + eta*(1817 + eta*(-425 + 9*eta))) + 35*eta*(4*(-4580*sqrt(1 - ipow(e,2)) + (745 + 600*sqrt(1 - ipow(e,2)) - 9*eta)*eta) + 123*(3 + sqrt(1 - ipow(e,2)))*ipow(M_PI,2)) + ipow(e,2)*(-6720*(62 + 35*sqrt(1 - ipow(e,2))) + eta*(8*(54239 + 20370*sqrt(1 - ipow(e,2)) + 105*eta*(-119 - 16*sqrt(1 - ipow(e,2)) + 3*eta)) + 4305*ipow(M_PI,2))))*ipow(x,3))/(6720.*ipow(-1 + ipow(e,2),2)) - ((2 + q)*x*sqrt(x)*Xi)/sqrt(1 - ipow(e,2)));
    
    const double r = ar*(1-er*cos(u));
    
    return r;
}

NEW_MODEL(Spin, "Spin");
