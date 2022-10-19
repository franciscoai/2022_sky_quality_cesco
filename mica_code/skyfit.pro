FUNCTION MOD_CURVEFIT,X,Y,W,A,SIGMAA, Function_Name = Function_Name, $
      STRANGE=strange
;+
; NAME:
;	MOD_CURVEFIT
;
; PURPOSE:
; Modified from the original curvefit function, since there
; occur endless loops sometimes...
;	Non-linear least squares fit to a function of an arbitrary 
;	number of parameters.  The function may be any non-linear 
;	function where the partial derivatives are known or can be 
;	approximated.
;
; CATEGORY:
;	E2 - Curve and Surface Fitting.
;
; CALLING SEQUENCE:
;	Result = CURVEFIT(X, Y, W, A, SIGMAA, FUNCTION_NAME = name)
;
; INPUTS:
;	X:  A row vector of independent variables.
;
;	Y:  A row vector of dependent variable, the same length as x.
;
;	W:  A row vector of weights, the same length as x and y.
;		For no weighting,
;		w(i) = 1.0.
;		For instrumental weighting,
;		w(i) = 1.0/y(i), etc.
;
;	A:  A vector, with as many elements as the number of terms, that 
;	    contains the initial estimate for each parameter.  If A is double-
;	    precision, calculations are performed in double precision, 
;	    otherwise they are performed in single precision.
;
; KEYWORDS:
;	FUNCTION_NAME:  The name of the function (actually, a procedure) to 
;	fit.  If omitted, "FUNCT" is used. The procedure must be written as
;	described under RESTRICTIONS, below.
;
; OUTPUTS:
;	Returns a vector of calculated values.
;	A:  A vector of parameters containing fit.
;
; OPTIONAL OUTPUT PARAMETERS:
;	Sigmaa:  A vector of standard deviations for the parameters in A.
;	
; COMMON BLOCKS:
;	NONE.
;
; SIDE EFFECTS:
;	None.
;
; RESTRICTIONS:
;	The function to be fit must be defined and called FUNCT,
;	unless the FUNCTION_NAME keyword is supplied.  This function,
;	(actually written as a procedure) must accept values of
;	X (the independent variable), and A (the fitted function's
;	parameter values), and return F (the function's value at
;	X), and PDER (a 2D array of partial derivatives).
;	For an example, see FUNCT in the IDL User's Libaray.
;	A call to FUNCT is entered as:
;	FUNCT, X, A, F, PDER
; where:
;	X = Vector of NPOINT independent variables, input.
;	A = Vector of NTERMS function parameters, input.
;	F = Vector of NPOINT values of function, y(i) = funct(x(i)), output.
;	PDER = Array, (NPOINT, NTERMS), of partial derivatives of funct.
;		PDER(I,J) = DErivative of function at ith point with
;		respect to jth parameter.  Optional output parameter.
;		PDER should not be calculated if the parameter is not
;		supplied in call.
;
; PROCEDURE:
;	Copied from "CURFIT", least squares fit to a non-linear
;	function, pages 237-239, Bevington, Data Reduction and Error
;	Analysis for the Physical Sciences.
;
;	"This method is the Gradient-expansion algorithm which
;	combines the best features of the gradient search with
;	the method of linearizing the fitting function."
;
;	Iterations are performed until the chi square changes by
;	only 0.1% or until 20 iterations have been performed.
;
;	The initial guess of the parameter values should be
;	as close to the actual values as possible or the solution
;	may not converge.
;
; MODIFICATION HISTORY:
;	Written, DMS, RSI, September, 1982.
;	Does not iterate if the first guess is good.  DMS, Oct, 1990.
;	Added CALL_PROCEDURE to make the function's name a parameter.
;		(Nov 1990)
;	12/14/92 - modified to reflect the changes in the 1991
;		   edition of Bevington (eq. II-27) (jiy-suggested by CreaSo)
; 29-JUN-1997: Endless loop ended by introducing a maxloop variable
;-
ON_ERROR,2		;RETURN TO CALLER IF ERROR
				;Name of function to fit
if n_elements(function_name) le 0 then function_name = "FUNCT"
strange=0

A = 1.*A		;MAKE PARAMS FLOATING
NTERMS = N_ELEMENTS(A)	;# OF PARAMS.
NFREE = (N_ELEMENTS(Y)<N_ELEMENTS(X))-NTERMS ;Degs of freedom
IF NFREE LE 0 THEN STOP,'Curvefit - not enough data points.'
FLAMBDA = 0.001		;Initial lambda
DIAG = INDGEN(NTERMS)*(NTERMS+1) ;SUBSCRIPTS OF DIAGONAL ELEMENTS
;
FOR ITER = 1,20 DO BEGIN	;Iteration loop
;
;		EVALUATE ALPHA AND BETA MATRICIES.
;
	CALL_PROCEDURE, Function_name,X,A,YFIT,PDER ;COMPUTE FUNCTION AT A.
	BETA = (Y-YFIT)*W # PDER
	ALPHA = TRANSPOSE(PDER) # (W # (FLTARR(NTERMS)+1)*PDER)
;
	CHISQ1 = TOTAL(W*(Y-YFIT)^2)/NFREE ;PRESENT CHI SQUARED.
				; If a good fit, no need to iterate
	if chisq1 lt total(abs(y))/1e7/NFREE then begin
		sigmaa = fltarr(nterms)	;Return all 0's
		return, yfit
	endif
;
;	INVERT MODIFIED CURVATURE MATRIX TO FIND NEW PARAMETERS.
;
  loopcount=0 & maxloopcount=10 ; added by AE, 29-JUN-1997
	REPEAT BEGIN
    loopcount=loopcount+1
		C = SQRT(ALPHA(DIAG) # ALPHA(DIAG))
		ARRAY = ALPHA/C
		ARRAY(DIAG) = ARRAY(DIAG)*(1.+FLAMBDA)		
		ARRAY = INVERT(ARRAY)
		B = A+ ARRAY/C # TRANSPOSE(BETA) ;NEW PARAMS
		CALL_PROCEDURE, Function_name,X,B,YFIT	;EVALUATE FUNCTION
		CHISQR = TOTAL(W*(Y-YFIT)^2)/NFREE ;NEW CHISQR
		FLAMBDA = FLAMBDA*10.	;ASSUME FIT GOT WORSE
;	ENDREP UNTIL CHISQR LE CHISQ1       ; Modified by AE, 29-JUN-1997
	ENDREP UNTIL ((CHISQR LE CHISQ1) OR (loopcount GE maxloopcount))
  IF (loopcount GE maxloopcount) THEN strange=1
;
	FLAMBDA = FLAMBDA/100.	;DECREASE FLAMBDA BY FACTOR OF 10
	A=B			;SAVE NEW PARAMETER ESTIMATE.
;	PRINT,'ITERATION =',ITER,' ,CHISQR =',CHISQR
;	PRINT,A
	IF ((CHISQ1-CHISQR)/CHISQ1) LE .001 THEN GOTO,DONE ;Finished?
ENDFOR			;ITERATION LOOP
;
message, 'Failed to converge', /INFORMATIONAL
;
DONE:   SIGMAA = SQRT(ARRAY(DIAG)/ALPHA(DIAG)) ;RETURN SIGMA'S
RETURN,YFIT		;RETURN RESULT
END

PRO fit_exp,x,a,f,pder
n=N_ELEMENTS(x)
f=a(0)+a(1)*EXP(x/a(2))
pder=FLTARR(n,3)
pder(*,0)=1
pder(*,1)=EXP(x/a(2))
pder(*,2)=-(a(1)*x*pder(*,1))/(a(2)^2)
RETURN
END

;###############################################################################
;+
; NAME:  
;	SKYFIT
;
; PURPOSE:
;	To describe the radial sky profile in the flatfielded images
;
; CATEGORY:
;	PICO-MICA
;
; CALLING SEQUENCE:
;	result=SKYFIT(image,header[, COEFF=coeff])
;
; INPUTS:
;	image:  The flatfielded image array
;       header: the corresponding image header.
;
; OPTIONAL INPUTS:
;	None
;
; KEYWORD PARAMETERS:
;	COEFF:  Returns the coefficients:
;               coeff(0) gives the sky brightness extrapolated
;                        to a place far from the occulter.
;               coeff(1) gives the amplitude of the fit,
;                        i.e. the brightness of the fit at
;                        the solar surface minus the brightness
;                        at infinite distance
;               coeff(2) gives the exponential damping factor of the
;                        outer fit (in Pixels)
;
;               Thus: the fit is: 
;                        coeff(1)*exp(x/coeff(2))+coeff(0)
;               and coeff(0)+coeff(1) gives the brightness at the
;               solar surface.
;
; OUTPUTS:
;	result: The fitted sky brightness profile, starting at the
;	        solar surface (i.e. at 1.0 Rs)
;
; OPTIONAL OUTPUTS:
;	None
;
; EXAMPLE:
;	PLOT, SKYFIT(image,header,COEFF=c)
;	
; COMMON BLOCKS:
;	None
;
; SIDE EFFECTS:
;	Unknown
;
; RESTRICTIONS:
;	The hole of the image should be calculated in the header
;
; PROCEDURE:
; Straightforward. An exponential fit is done to a radial cut
;   in the image, by default from the center to the right edge.
;   The parameters describe the sky brightness well and are
;   returned in the keyword.
;
; MODIFICATION HISTORY:
;	V1.0 Alexander Epple, MPAe Lindau, 12-JUN-1996
; V2.0 AE, El Leoncito, 28-JUN-1997: much more simplified
;       version for standard image treatment. The original
;       function is renamed in STRAYFIT.pro
;-

FUNCTION skyfit,image,header, COEFF=coeff, OVEROCC=overocc

IF (N_ELEMENTS(overocc) EQ 0) THEN overocc=40
sz=SIZE(image)

hole=[SXPAR(header,'HOLE0'),SXPAR(header,'HOLE1'), $
      SXPAR(header,'HOLE2'),SXPAR(header,'HOLE3')]
IF (TOTAL(ABS(hole)) EQ 0) THEN hole=GET_HOLE(image)
IF (TOTAL(ABS(hole)) EQ 0) THEN hole=[sz(1:2)/2,520,520]

; Take the western half cut
x0=hole(0)+hole(2)/2
x1=sz(1)-1
cut=FLOAT(REBIN(image(x0:x1,hole(1)+[-1,0,1]),x1-x0+1,1))

; Don't take the inner overocc pixels
cut=cut(overocc:*)
n=N_ELEMENTS(cut)

limb_to_occultor=0.05*MEAN(hole(2:3))/2.1
overocc=overocc+limb_to_occultor

; now do the fit of the exponential
x=FIX(INDGEN(n)+overocc)

; do initial guesses for the fitting parameters: take values at 3 points
coeff=FLTARR(3)

i1=0 & i2=n/2 & i3=n-1

coeff(0)=cut(i3)/2
dummy=(cut(i1)-coeff(0))/(cut(i2)-coeff(0))
dummy=ALOG(dummy)
coeff(2)=(x(i1)-x(i2))/dummy
coeff(1)=(cut(i1)-coeff(0))/EXP(x(i1)/coeff(2))

result=MOD_CURVEFIT(x, $
                    cut, $
                    REPLICATE(1.,N_ELEMENTS(cut)), $
                    coeff, $
                    FUNCTION_NAME='fit_exp', $
                    STRANGE=strange)
IF KEYWORD_SET(strange) THEN BEGIN
  MESSAGE,/INFO,/CONT,'The fit did not converge!'
  coeff=0
  result(*)=0
ENDIF

RETURN,result

END
