import numpy as np

def initial_values():
    ## LIST OF INITIAL CONDITIONS.
    # Name     Value     Index     Units
    Glu0  =  39.7;       # 1       g/L
    Xyl0  =  23.5;       # 2       g/L
    Fur0  =  0.56;       # 3       g/L
    FA0   =  0;          # 4       g/L
    HMF0  =  0.2;        # 5       g/L
    HAc0  =  3.05;       # 6       g/L
    # Ac0   =  0.0001;    # 7       g/L
    EtOH0 =  0.62;       # 7       g/L
    X0    =  1.75;       # 8       g/L
    
    init = [Glu0, Xyl0, Fur0, FA0, HMF0, HAc0, EtOH0, X0]
    return init


def kinetics(t,x,par):
    #  The following 8 components are considered in the model: 
    #  0. Glucose
    #  1. Xylose
    #  2. Furfural
    #  3. Furfuryl alcohol
    #  4. 5-HMF
    #  5. HAc
    #  6. Ethanol
    #  7. Biomass

    #  According to the kinetic model, the following chemical equations describe
    #  the system. 

    #  1. Glucose  -----> Ethanol + Biomass
    #  2. Xylose   -----> Ethanol + Biomass
    #  3. Furfural -----> Furfuryl alcohol 
    #  4. HMF      -----> Acetic acid
    #  5. HAc     <-----> Ac- [optional - not used here]
    #  6. HAc      -----> Biomass (maintenence) 


    ## 1. STOICHIOMETRIC MATRIX. 
    # The 8 components and 6 reactions are considered in this model. One could
    # also expand the model with Ac-, but that is not implemented.

    #                Glu  Xyl  Fur  FA           HMF  HAc           EtOH        X  
    st = np.array([[-1,   0,   0,   0,           0,   0,            par.YPSg,   par.YXSg],      # Glucose uptake
                   [ 0,  -1,   0,   0,           0,   0,            par.YPSx,   par.YXSx],      # Xylose uptake
                   [ 0,   0,  -1,   par.Y_FA_Fur,0,   0,            0,          0       ],      # Furfural uptake
                   [ 0,   0,   0,   0,          -1,   par.Y_HAc_HMF,0,          0       ],      # HMF uptake
                   [ 0,   0,   0,   0,           0,  -1,            0,          0       ]])     # HAc uptake




    ## 2. DEFINE INDIVIDUAL KINETICS AND INHIBITION TERMS. 
    # Ph stands for phenomena, and it includes all the phenomena considered in
    # the model, including reaction, inhibition and equilibria. Each of the
    # phenomena is implemented separately below, for clarity
    # 1. GLUCOSE UPTAKE RATE.
    # numaxG: max consumption rate of glucose           s^-1       [1]
    # KSPG:   affinity constant glucose                 g/L        [1]
    # KiPG:   inhibition constant glucose               g/L        [1]
    Ph1 = par.numaxG * x[0] * x[7] / (par.KSPG + x[0] + (x[0]**2)/par.KiPG)
    Ph1 = max(0,Ph1)
    # 2. XYLOSE UPTAKE RATE.
    # numaxX: max consumption rate of xylose            s^-1       [1]
    # KSPX:   affinity constant xylose                  g/L        [1]
    # KiPX:   inhibition constant xylose                g/L        [1]
    Ph2 = par.numaxX * x[1] * x[7] / (par.KSPX + x[1] + (x[1]**2)/par.KiPX)
    Ph2 = max(0,Ph2)

    # 3. FURFURAL UPTAKE RATE.
    # numaxFur: max uptake rate of furfural               s^-1       [2]
    # KSFur:    affinity constant furfural                g/L        [2]
    Ph3 = par.numaxFur * x[2] * x[7] / (par.KSFur + x[2])
    Ph3 = max(0,Ph3)

    # 4. FURFURAL CONVERSION INTO FURFURYL ALCOHOL.
    # Y_FA_Fur:  yield coefficient FA/Fur                  gFA/gFur   [2]
    Ph4 = Ph3 * par.Y_FA_Fur
    Ph4 = max(0,Ph4)

    # 5. FURFURYL ALCOHOL INHIBITS GLUCOSE UPTAKE RATE.
    # KiFAg:    inhibition constant of FA on Glu uptake   g/L        [2]
    Ph5 = 1 / (1 + x[3]/par.KiFAg)
    Ph5 = max(0,Ph5)

    # 6. FURFURYL ALCOHOL INHIBITS XYLOSE UPTAKE RATE.
    # KiFAx:    inhibition constant of FA on Xyl uptake   g/L        [2]
    Ph6 = 1 / (1 + x[3]/par.KiFAx)
    Ph6 = max(0,Ph6)

    # 7. FURFURAL INHIBITS GLUCOSE UPTAKE RATE.
    # KiFurg:   inhibition constant of Fur on Glu uptake  g/L        [2]
    Ph7 = 1 / (1 + x[2]/par.KiFurg)
    Ph7 = max(0,Ph7)

    # 8. FURFURAL INHIBITS XYLOSE UPTAKE RATE.
    # KiFurx:   inhibition constant of Fur on Xyl uptake  g/L        [2]
    Ph8 = 1 / (1 + x[2]/par.KiFurx)
    Ph8 = max(0,Ph8)

    # 9. FURFURAL INHIBITS HMF UPTAKE RATE.
    # KiFurHMF: inhibition constant of Fur on HMF         g/L        [2]
    Ph9 = 1 / (1 + x[2]/par.KiFurHMF)
    Ph9 = max(0,Ph9)

    # 10. HMF INHIBITS GLUCOSE UPTAKE RATE.
    # KiHMFg:   inhibition constant of HMF on Glu         g/L        [2]
    Ph10 = 1 / (1 + x[4]/par.KiHMFg)
    Ph10 = max(0,Ph10)

    # 11. HMF INHIBITS XYLOSE UPTAKE RATE.
    # KiHMFx:   inhibition constant of HMF on Xyl         g/L        [2]
    Ph11 = 1 / (1 + x[4]/par.KiHMFx)
    Ph11 = max(0,Ph11)

    # 12. HMF UPTAKE RATE. 
    # numaxHMF: max uptake rate of HMF                    s^-1       [2]
    # KSHMF:    affinity constant HMF                     g/L        [2]
    Ph12 = par.numaxHMF * x[7] * x[4] / (par.KSHMF + x[4])
    Ph12 = max(0,Ph12)

    #  13. pH INFLUENCES THE PAIR HAc/Ac.
    # pH = 5.5;              % pH is controlled
    # pKa = 4.75;            % pKa of acetic acid.
    # Ph13 = x(7,1) / x(6,1) * 1 / (1 + 10^(-pH)/10^(-pKa));
    # Ph13 = max(0,Ph13);


    # 14. HAc UPTAKE RATE.
    # numaxHAc: max uptake rate of HAc                    s^-1       [2]
    # KSHAc:    affinity constant HAc                     g/L        [2]
    Ph14 = x[7] * x[5] * par.numaxHAc / (par.KSHAc + x[5])
    Ph14 = max(0,Ph14)


    #15. HMF CONVERSION INTO HAc
    #Y_HAc_HMF = p(21);     % yield coefficient HAc/HMF                 gHAc/gHMF  [2], [3]
    #Ph15 = Y_HAc_HMF * Ph12;
    #Ph15 = max(0,Ph15);

    # 16. HAc INHIBITS GLUCOSE UPTAKE RATE.
    # KiHAcg:   inhibition constant of HAc on Glu         g/L        [2]
    Ph16 = 1 / (1 + x[5]/par.KiHAcg)
    Ph16 = max(0,Ph16)

    # 17. HAc INHIBITS XYLOSE UPTAKE RATE.
    # KiHAcx:   inhibition constant of HAc on Xyl         g/L         [2]
    Ph17 = 1 / (1 + x[5]/par.KiHAcx)
    Ph17 = max(0,Ph17)

    #  18. PRODUCTION OF ETHANOL FORM GLUCOSE AND XYLOSE.
    # YPSg:     yield ethanol-glucose                     gEtOH/gGlu [1]
    # YPSx:     yield ethanol-xylose                      gEtOH/gGlu [1]
    # Ph18a = par.YPSg * Ph1;
    # Ph18b = par.YPSx * Ph2;
    # Ph18a = max(0,Ph18a);
    # Ph18b = max(0,Ph18b);

    # 19. ETHANOL INHIBITS THE UPTAKE OF GLUCOSE AND XYLOSE.
    # PMPg:     inhibition constant for glucose           g/L        [1]
    # gammaG:   exponent factor inhibition glucose        g/L        [1]
    # PMPx:     inhibition constant for xylose            g/L        [1]
    # gammaX:   exponent factor inhibition xylose         g/L        [1]
    Ph19a = 1-(x[6]/par.PMPg)**par.gammaG
    Ph19b = 1-(x[6]/par.PMPx)**par.gammaX
    Ph19a = max(0,Ph19a)
    Ph19b = max(0,Ph19b)

    #  20. CELL GROWTH.
    # mGlu:     maintenance constant from glucose         g/L        [1]
    # mXyl:     maintenance constant from xylose          g/L        [1]
    # YXSg:     yield X-Glu                               gX/gGlu    [1]
    # YXSx:     yield X-Xyl                               gX/gXyl    [1]
    # mumaxG:   max growth of X from Glu                  h-1        [1]
    # mumaxX:   max frowth of X from Xyl                  h-1        [1]

    # Ph20a = max(0,(Ph1 + mGlu*x(8,1))*YXSg);
    # Ph20b = max(0,(Ph1 + mXyl*x(8,1))*YXSx);

    # 21. CATABOLITE REPRESSION.
    # KiGlu:    inhibition constant of Glu to Xyl         g/L        Unknown
    Ph21 = 1 / (1 + x[0]/par.KiGlu)

    # 22. ACETATE IS USED FOR MAINTENANCE.
    # mHAc:     maintenance constant from HAc              g/L        Unknown
    # YXSHAc:   yield acetate biomass                      g/L        Unknown
    Ph22 = max(0,par.mHAc * x[7] * par.YXSHAc)

    ## 3. REACTION RATES. 
    # This includes the reaction rates with inhibition terms.
    rates = np.zeros(5)
    rates[0]  = Ph1 * Ph5 * Ph7 * Ph10 * Ph16 * Ph19a                            # Glucose uptake rate considering inhibitions.
    rates[1]  = Ph2 * Ph6 * Ph8 * Ph11 * Ph17 * Ph19b * Ph21                     # Xylose uptake rate considering inhibitions.
    rates[2]  = Ph3                                                              # Furfural uptake rate.
    rates[3]  = Ph12 *  Ph9                                                      # 5-HMF uptake rate considering inhibitions.
    rates[4]  = Ph14
    

    dxdt = np.zeros(8)
    r, c = st.shape
    for i in range(c):
        dxdt[i] = sum([rates[j] * st[j,i] for j in range(r)])
    ## 4. MASS BALANCE FOR THE SYSTEM.
    # dxdt = (rr'*st)'
    #  dxdt = dxdt';
    return dxdt

if __name__=='__main__':
    from parameters import Parameters
    init = initial_values()
    par = Parameters()
    dxdt = kinetics(0, init, par)
    print(dxdt)



