function flarefac, image, VLIMB=vlimb, FITLIMB=fitlimb

;+
;NAME
;   flarefactor
;
;PURPOSE:
;
; This routine extract a merit factor from the sun image for
; evaluate if any flare appear on the sun disk.
;
;CALLING SEQUENCE:
;
;	ff=flarefactor(image, VLIMB=700)
;
;CALIFICATION
;
; HASTA- Routine
;
;INPUT:
;  image = the image frame adquired by the CCD camera
;
;KEYWORDS:
;
;   VLIMB = the guest value for the intensity on the limb
;           in ADUs. If this value is not present, the program
;           sets this value as 700 ADUs (the intensity value
;           of the limb with time exposure of 100 ms).
;
;   FITLIMB = if this keyword is set, then the program fit the
;           intensity value on the limb in ADUs. This value is
;           used to define low limit of intensities to consider
;           in the process
;
;KEYWORDS:

if n_params() eq 0 then begin
	Ff=-1
    goto, fin
endif

iter=0

if not KEYWORD_SET(vlimb) then vlimb=700 ; default value of intensity on the limb

if KEYWORD_SET(fitlimb) then iter=1

if max(image) lt vlimb then begin
    Ff=-1
    goto, fin
endif

Histovec=histogram(image,min=vlimb,max=4095) ; 4095 max value in ADUs

tot=TOTAL(Histovec)
Xs=size(Histovec)
Xvec=indgen(Xs(1))+vlimb
Yvec=Histovec/tot
NUM=N_ELEMENTS(Xvec)
mu=double(Xvec##TRANSPOSE(Yvec)) ; the arithmetic mean

if iter gt 0 then begin

   darklimb_function, lambda=6563.5, bandpass=0.3, limbfilt=dlimb

   muold(0)=0.

   desv=(sqrt((mu(0)-muold(0))^2)/mu(0))*100
   error=0.05

   while (desv gt error) do begin

		dark=dlimb*mu(0)
		vlimb=round(min(Dark))

		Histovec=histogram(image,min=vlimb,max=4095)

		tot=TOTAL(Histovec)
		Xs=size(Histovec)
		Xvec=indgen(Xs(1))+vlimb
		Yvec=Histovec/tot
		NUM=N_ELEMENTS(Xvec)
		muold=mu
		mu=double(Xvec##TRANSPOSE(Yvec)) ; the arithmetic mean

		desv=(sqrt((mu(0)-muold(0))^2)/mu(0))*100



   endwhile


endif



dif2=dblarr(NUM)
for k=0,NUM-1 do dif2(k)=(Xvec(k)-mu)^2
vs=double(dif2##TRANSPOSE(Yvec)) ; the variance
ds=sqrt((tot*vs(0))/(tot-1)) 		 ; Universe Standard deviation

;ijs=round(mu(0))+round(ds) ; The intensity at 1 desviation of mean value

ijl=where(Histovec gt 0,njl)
;if ijl(njl-1) le mu(0) then print, 'mu:',mu(0),'ijl max:',ijl(njl-1)

Ff=(((ijl(njl-1)+vlimb)-mu(0))/(ds*3.))*100. ; factor de merito

fin:

return, Ff(0)

end
