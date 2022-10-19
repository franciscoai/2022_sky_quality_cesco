FUNCTION flatting, imagenr, flatnr, $
                   FACTOR=factor, $
                   FLAT2REFSUN=flat2refsun, $
                   FLATADD=flatadd, $
                   FLATHDR=flatheader, $
                   FLATHISTORY=flathistory, $
                   GREENFLATCORR=greenflatcorr, $
                   GRID=grid, $
                   HOLE=khole, $
                   IMAGEADD=imageadd, $
                   IMAGEHDR=imageheader, $
                   LEVEL=level,  $
                   ANNOTATE=annotate, $
                   NOFINALCENTER=nofinalcenter, $
                   HOLEZERO=holezero, $
                   NOKEEPFLAT=nokeepflat, $
                   NOKEEPIMAGE=nokeepimage, $
                   OCTTREAT=octtrt, $
                   QUIET=quiet, $
                   REGION=region, $
                   SAVE=ksave,  $
                   SHIFT=kshft, $
                   VERBOSE=verbose, $
                   WORKDIR=workdir

;+
; NAME:
;	FLATTING
;
; PURPOSE:
;	Perform the flatfielding of an image with the
;	corresponding flat.
;
; CATEGORY:
;	PICO
;
; CALLING SEQUENCE:
;	result=FLATTING(image,flat)
;
; INPUTS:
;	image:  The image which should be flatted. Either
;		a two dimensional array containig the image
;		or just the serial number can be entered.
;		In the second case the image corresponding
;		to that number is fetched by GET_NR. See
;		there for any restrictions.
;	flat:   The corresponding flat. Also here, either
;		a two dimensional array containing the flat
;		or just the serial number of the flat can
;		be specified.
;
; OPTIONAL INPUT PARAMETERS:
;	None
;
; KEYWORD PARAMETERS:
;	HOLE:   Radiusvector of the hole in the form
;		[xcenter, ycenter, xdiam, ydiam]. This format
;		is compatible with the result of GET_HOLE.
;		If the hole is specified, it is set to 0
;	SAVE:   If set and if the image is passed by serial
;		number, or if the imageheader is passed by
;		IMAGEHDR, the image will be saved in the directory
;		specified by WORKDIR or by default in d:\temp.
;		Alike the name convention the name is
;		nnnnFfpe.int, where nnnn gives the serial
;		number of the image, F denotes 'flatfielded',
;		f, p and e represent the filter, polarizer and
;		exposure time as in the original filename.
;		Default: no saving
;	FLATHISTORY: A string can be specified to be written in
;		the header of the treated image under 'history'.
;		By convention this string should begin with
;		'Flatfield: ' and should contain informations about
;		the used flatfield(s).
;	LEVEL:  Level (in fractions) at which the image should
;		be cut by CUT_MAX. The cutting will be performed
;		after the hole was set to 0!
;		Default: No cutting performed. If a value is
;		chosen, it is adviced to be around 0.98 or so.
;	SHIFT:  a two element vector in the form [xshft, yshft]
;		to shift the flatfield with respect to the image
;		Convention: a shift of the flatfield to the upper
;		right (increasing pixel values) is performed if
;		xshft and yshft both are positiv. Take care of this
;   since the flatfield then does not fit any more to the
;   image pixel by pixel!!!
;		By default there is no shift.
;	NOFINALCENTER: If set, the hole in the image will NOT be
;		recentered after flatfielding; by default the
;		center of the hole is centered on [xdim,ydim]/2
;		by calling IMAGESHIFT
;	NOKEEPFLAT: If set, the original entry flat is NOT
;		kept in memory but deleted on call. This
;		saves memory in case of memory problems
;		(the swapping problem of IDL!)
;	NOKEEPIMAGE: If set, the original entry image is NOT
;		kept in memory but deleted on call. This
;		saves memory in case of memory problems
;		(the swapping problem of IDL!)
;	IMAGEADD, FLATADD: A constant to be added to the
;		whole image resp. flatfield before flatfielding.
;		and AFTER the hole of the image is set to zero.
;		Default: IMAGEADD=0, FLATADD=0
;	IMAGEHDR: The header of the image (optional)
;	FLATHDR: The header of the flat (optional)
;		before flatfielding.
;	FACTOR: upon return returns the scaling factor of the
;		flat
;	NOHOLEZERO: If set, the hole neither of the image nor
;		of the flat will be set to zero. Default: The
;		mean value of the holes will be subtracted from
;		the images before division.
;	ANNOTATE: If set, the major header entries will be written
;		in the holecenter.
; GRID:   If set, a coordinate grid is drawn in the center
;   of the hole
;	REGION: a four element vector specifying the region
;		in the image and the flat to determine the factor of
;   multiplication. Default: The normal gauge region
;	VERBOSE: If set, the duration of the flatfielding is
;		displayed in seconds
;
; OUTPUTS:
;	result: The flatfielded image.
;
; OPTIONAL OUTPUT PARAMETERS:
;	None
;
; COMMON BLOCKS:
;	None
;
; SIDE EFFECTS:
;	An already existing file with the same name as the
;	flatfielded image might be overwritten.
;
; RESTRICTIONS:
;	None
;
; EXAMPLE:
;	To divide an image by its flatfield and to save it on
;	the temp disk enter:
;
;	image=flatting(2569,flat,/SAVE,FLATHISTORY='Flatfield $
;		MEAN(2573, 2583, 2593)')
;
; PROCEDURE: Straightforward.
;
; MODIFICATION HISTORY:         ae 11-AUG-1994, Pic Du Midi
; 	time optimized for Pic Du Midi 8-MAR-1994 Pic Du Midi
;		(splitting of flatdivision in subdivisions)
;	NOKEEPIMAGE added: 28-SEP-1995
;	GREENFLATCORR added: 07-DEC-1995
;-

time=systime(1)
quiet=1

;*********************************************************************
; Prepare the keywords and set default values
;*********************************************************************

IF (N_ELEMENTS(kshft) EQ 0) THEN shft=[0,0] ELSE shft=kshft
IF (N_ELEMENTS(holezero) EQ 0) THEN holezero=0
IF (N_ELEMENTS(annotate) EQ 0) THEN annotate=0
IF (N_ELEMENTS(nokeepflat) EQ 0) THEN nokeepflat=0
IF (N_ELEMENTS(nokeepimage) EQ 0) THEN nokeepimage=0
IF (N_ELEMENTS(flip) EQ 0) THEN flip=0
IF (N_ELEMENTS(octtrt) EQ 0) THEN octtrt=0
IF (N_ELEMENTS(workdir) EQ 0) THEN workdir=!PICO.WORKDIR
IF (N_ELEMENTS(khole) NE 4) THEN hole=0 ELSE hole=khole
IF (N_ELEMENTS(nofinalcenter) EQ 0) THEN nofinalcenter=0
IF (N_ELEMENTS(ksave) EQ 0) THEN save=0 ELSE BEGIN
  save=1
  savenr=ksave
ENDELSE

IF (N_ELEMENTS(flathistory) GT 0) THEN BEGIN
  IF (flathistory(0) EQ '') THEN UNDEFINE,flathistory
ENDIF

overflownvalue=32000	; the value of the pixels which have been
			; overflown in the original image (has to be
			; conform with T_CONT)

;*********************************************************************
; Get the image
;*********************************************************************

t=SYSTIME(1)
		; if imagenr is NOT an array interprete it as a number to
		; be fetched on the disk. The headers can be fetched as
		; well. SAVE is set to the imagenumber.

IF (N_ELEMENTS(imagenr) EQ 1) THEN BEGIN

  imagename=FULL_NAME(imagenr)
  IF (STRUPCASE(imagename) NE 'ERROR') THEN BEGIN
  	imageheader=READ_HDR(imagename)
	  image=REPAIR(GET_IMG(imagename, QUIET=1-KEYWORD_SET(verbose)),imageheader)
  	savenr=imagenr
  ENDIF ELSE BEGIN
	  MESSAGE,'There is no image with number '+SC(imagenr)
  ENDELSE
ENDIF ELSE BEGIN
  image=imagenr
  IF (N_ELEMENTS(imageheader) GT 0) THEN BEGIN
	FNHANDLE,HEADER=imageheader,NUMBER=savenr
    ENDIF ELSE BEGIN
	IF (KEYWORD_SET(save)) THEN BEGIN
	    MESSAGE,/INFO,/CONT,'To save: specify imageheader! NO SAVE'
	    save=0
	ENDIF
    ENDELSE
ENDELSE

IF (N_ELEMENTS(imageheader) EQ 0) THEN BEGIN
    save=0
    imageheader=DEF_HDR()
ENDIF

IF (DATA_TYPE(imageheader) EQ 7) THEN fitsheader=1 ELSE fitsheader=0

IF KEYWORD_SET(nokeepimage) THEN BEGIN
    IF (N_ELEMENTS(imagenr) GT 1) THEN UNDEFINE,imagenr
ENDIF

IF (NOT KEYWORD_SET(quiet)) THEN BEGIN & print,'image read '+SC(SYSTIME(1)-t,DEC=2) & FLUSH,-1 & t=SYSTIME(1) & ENDIF

;*********************************************************************
; Get the flatfield
;*********************************************************************

IF (N_ELEMENTS(flatnr) EQ 1) THEN BEGIN

    flatname=FULL_NAME(flatnr)
    IF (STRUPCASE(flatname) NE 'ERROR') THEN BEGIN
	flatheader=READ_HDR(flatname)
	flat=REPAIR(GET_IMG(flatname, QUIET=1-KEYWORD_SET(verbose)),flatheader)
    ENDIF ELSE BEGIN
	MESSAGE,'There is no flat with number '+SC(imagenr)
    ENDELSE
ENDIF ELSE flat=flatnr

IF (N_ELEMENTS(flatheader) EQ 0) THEN flatheader=DEF_HDR()

IF KEYWORD_SET(nokeepflat) THEN BEGIN
    IF (N_ELEMENTS(flatnr) GT 1) THEN UNDEFINE,flatnr
ENDIF

IF (N_ELEMENTS(flathistory) GT 0) THEN BEGIN
    ADDHISTORY,imageheader,flathistory
ENDIF ELSE BEGIN
  IF (N_ELEMENTS(flatheader) NE 0) THEN BEGIN
    FNHANDLE,HEADER=flatheader,NUMBER=flatnumber
    ADDHISTORY,imageheader,'FLATTING: Flatfield '+STRCOMPRESS(flatnumber,/REMOVE_ALL)
  ENDIF ELSE ADDHISTORY,imageheader,'FLATTING: Flatfield: ?'
ENDELSE

IF (NOT KEYWORD_SET(quiet)) THEN BEGIN & print,'flat read '+SC(SYSTIME(1)-t,DEC=2) & FLUSH,-1 & t=SYSTIME(1) & ENDIF

overflownindex=WHERE(image EQ 4095,overflowncount)

sz=SIZE(image)
szx=sz(1)
szy=sz(2)

IF (NOT KEYWORD_SET(quiet)) THEN BEGIN & print,'size and overflown pixels determined '+SC(SYSTIME(1)-t,DEC=2) & FLUSH,-1 & t=SYSTIME(1) & ENDIF

;*********************************************************************
; Check whether image and flat fit together (Can only be done if
; the two headers are available)
;*********************************************************************

IF ((N_ELEMENTS(imageheader) NE 0) AND (N_ELEMENTS(flatheader) NE 0)) THEN BEGIN
  IF KEYWORD_SET(fitsheader) THEN BEGIN
    cond=((SXPAR(imageheader,'FILTER') NE SXPAR(flatheader,'FILTER')) OR $
        (SXPAR(imageheader,'POLAR') NE SXPAR(flatheader,'POLAR')))
  ENDIF ELSE BEGIN
    cond=((imageheader.filter NE flatheader.filter) OR $
		(imageheader.polarizer NE flatheader.polarizer))
  ENDELSE
  IF cond THEN BEGIN
	MESSAGE,/CONT,/INFO,/NONAME,'WARNING: The image and flat '+ $
		' don''t fit together'
  ENDIF
ENDIF

;*********************************************************************
; Get the hole of the flat (if not yet specified);
; Determine the region in the hole in which the images and flats
; shall be set to 0 ('center'). Either it is 20% smaller as the
; filling rectangle around the holecenter (if known) or it is a
; standard value.
;*********************************************************************

IF (N_ELEMENTS(hole) NE 4) THEN BEGIN
  IF KEYWORD_SET(fitsheader) THEN BEGIN
    hole=[SXPAR(flatheader,'HOLE0'),SXPAR(flatheader,'HOLE1'), $
          SXPAR(flatheader,'HOLE2'),SXPAR(flatheader,'HOLE3')]
  ENDIF ELSE hole=flatheader.hole
ENDIF
IF (TOTAL(ABS(hole)) EQ 0) THEN hole=GET_HOLE(flat)
IF (TOTAL(ABS(hole)) GT 0) THEN BEGIN

  IF KEYWORD_SET(fitsheader) THEN orient=SXPAR(imageheader,'ORIENT') ELSE orient=imageheader.orient
  ADDHISTORY,imageheader,'FLATTING: Original hole: ['+ $
  SC(hole(0))+','+SC(hole(1))+','+SC(hole(2))+','+SC(hole(3))+ $
  ']; orient: '+SC(orient)

; write the hole to the headers
  IF KEYWORD_SET(fitsheader) THEN BEGIN
    FOR i=0,N_ELEMENTS(hole)-1 DO SXADDPAR,imageheader,'HOLE'+SC(i),hole(i)
  ENDIF ELSE BEGIN
    imageheader.hole=hole
    flatheader.hole=hole
  ENDELSE

; determine a rectangular region in the center of the hole to set
; it to zero eventually

  dims=.85*(hole(2)+hole(3))/(2*SQRT(2))
  center=FIX(hole+dims*[-1,1,-1,1]/2)
ENDIF ELSE center=[sz(1),sz(2),sz(1),sz(2)]+[-1,1,-1,1]*150


;*********************************************************************
; The next two procedures only applies very specially to some
; PICO images and is of no general use!
;*********************************************************************
;---------------------------------------------------------------------
; If GREENFLATCORR is set, the upper left edge of the green line
; flats are corrected in order to remove parasite light.
;---------------------------------------------------------------------
IF (NOT KEYWORD_SET(fitsheader)) THEN BEGIN
  IF (KEYWORD_SET(greenflatcorr) AND $
	  (imageheader.filter EQ 5) AND $
  	(imageheader.polarizer EQ 1)) THEN BEGIN
    GFCORR,flat,HEADER=flatheader
    ADDHISTORY,imageheader,'Flat was treated by GFCORR prior to flatfielding'
  ENDIF
ENDIF

;---------------------------------------------------------------------
; If OCTTREAT is set, do the correction for parasit light in the
; image and the flat. This as well does only apply to early PICO-images
; of october 1993!
;---------------------------------------------------------------------

IF KEYWORD_SET(octtrt) THEN BEGIN
  OCTTREAT,IMAGE=image,FLAT=flat,FIT=OCTFIT(flat,HOLE=hole)
  IF (KEYWORD_SET(imageheader)) THEN BEGIN
  	ADDHISTORY,imageheader,'OCTTREAT: polynomial surface of degree 2 '+ $
                        'subtracted from flat and image'
  ENDIF
  holezero=0
ENDIF

;*********************************************************************
; set the holecenter to zero
;*********************************************************************

IF KEYWORD_SET(holezero) THEN BEGIN
  imagemean=FIX(MEAN(EXTRACT(IMAGE,center)))
  image=TEMPORARY(image)-imagemean

  flatmean=FIX(MEAN(EXTRACT(flat,center)))
  flat=TEMPORARY(flat)-flatmean

  IF KEYWORD_SET(fitsheader) THEN orient=SXPAR(imageheader,'ORIENT') ELSE orient=imageheader.orient
  ADDHISTORY,imageheader,'FLATTING: Mean value of ['+ $
    	SC(center(0))+':'+SC(center(1))+','+ $
	    SC(center(2))+':'+SC(center(3))+ $
      '], orient='+SC(orient)+', of img and flat set to zero'
ENDIF ELSE BEGIN
;  ADDHISTORY,imageheader,'FLATTING: Hole not set to 0 neither in image nor in flat'
ENDELSE

;*********************************************************************
; Add the constants imageadd and flatadd to image and flat
;*********************************************************************
IF (N_ELEMENTS(imageadd) GT 0) THEN BEGIN
  image=TEMPORARY(image)+FIX(imageadd+.5)
 	ADDHISTORY,imageheader,'FLATTING: A constant of '+SC(imageadd)+ $
  	' added to the image before flatfielding'
ENDIF

IF (N_ELEMENTS(flatadd) GT 0) THEN BEGIN
  flat=TEMPORARY(flat)+FIX(flatadd)
 	ADDHISTORY,imageheader,'FLATTING: A constant of '+SC(flatadd)+ $
  	' added to the flat before flatfielding'
ENDIF

;*********************************************************************
; factor to scale the image after division by the flatfield
;*********************************************************************

IF (N_ELEMENTS(region) GT 0) THEN BEGIN
  factor=MEAN(EXTRACT(flat,region))
ENDIF ELSE BEGIN
  factor=GAUGE(flat,REGION=region)
ENDELSE

IF KEYWORD_SET(fitsheader) THEN orient=SXPAR(imageheader,'ORIENT') ELSE orient=imageheader.orient
ADDHISTORY,imageheader,'FLATTING: Mean flatfield value '+SC(FIX(factor))+ $
	' in ['+SC(region(0))+','+SC(region(1))+','+ $
	SC(region(2))+','+SC(region(3))+']; orient='+SC(orient)

;*********************************************************************
; determine the correspondance of counts to brightness in solar units ppm.
; ppm=ct2ppm*counts of the final treated image. The exposure time
; is already taken into account, but NOT the filter Transmittance!!
;*********************************************************************

SXADDPAR,header,'PHOTOMET','CTS'

IF (N_ELEMENTS(Flat2Refsun) GT 0) THEN BEGIN
  IF (NOT KEYWORD_SET(fitsheader)) THEN BEGIN
    ct2ppm=REAL_EXPTIME(flatheader.exptime)/ $
           REAL_EXPTIME(imageheader.exptime)
  ENDIF ELSE BEGIN
    ct2ppm=SXPAR(flatheader,'EXPTIME')/ $
           SXPAR(imageheader,'EXPTIME')
  ENDELSE

  ct2ppm=1.E6*ct2ppm/factor
  ct2ppm=ct2ppm/Flat2Refsun

  IF (NOT KEYWORD_SET(fitsheader)) THEN BEGIN
    imageheader.ct2ppm=TEMPORARY(ct2ppm)
  ENDIF ELSE BEGIN
    SXADDPAR,imageheader,'CT2PPM',ct2ppm
  ENDELSE
ENDIF

;*********************************************************************
; shift the flatfield with respect to the image
; and fill the 'dead' columns and rows
;*********************************************************************

IF (KEYWORD_SET(shft)) THEN BEGIN
  IF (TOTAL(ABS(shft)) GT 0) THEN BEGIN
  	flat=IMAGESHIFT(TEMPORARY(flat),shft)
	  ADDHISTORY,imageheader,'FLATTING: Flat shifted by ['+ $
		  SC(shft(0))+','+SC(SHFT(1))+'] with respect to image'
  ENDIF
ENDIF

;************************************************************************************************
; ======================== WITH FEW MB of MEMORY ==============================

;*********************************************************************
; DOIT!!! in small subsections of s lines (to avoid memory problems)
;*********************************************************************
;;;s=16
;;;ny=szy/s

;;;t=SYSTIME(1)
;;;FOR j=0,ny-1 DO BEGIN
  ;;;IF KEYWORD_SET(verbose) THEN PRINT_POINT
  ;;;tf=flat(*,j*s:(j+1)*s-1)>1
  ;;;ti=image(*,j*s:(j+1)*s-1)
  ;;;image(*,j*s:(j+1)*s-1)=FIX(((factor*TEMPORARY(ti))/TEMPORARY(tf))>0<32767)	; flatfielding
;;;ENDFOR

;************************************************************************************************
; ======================== WITH TOO MUCH MB of MEMORY ==============================
  image=FIX(((factor*TEMPORARY(image))/TEMPORARY(flat>1))>0<32767)	; flatfielding
;************************************************************************************************

IF KEYWORD_SET(verbose) THEN print

UNDEFINE,flat
;---------------------------------------------------------------------
; extrapolate the left and top edge to get appropriate values for
; the noise therein (results still from the initial REPAIR and
; applies only to the PICO images)
;---------------------------------------------------------------------
;EXTRAPOL,image,3,21	; left edge
;EXTRAPOL,image,2,3	; top edge

;---------------------------------------------------------------------
; If there were overflown pixels in the original image set them back
; to a certain value given by overflownvalue (set in the beginning of
; the program). This mus be performed prior to the flipping
;---------------------------------------------------------------------

IF (overflowncount GT 0) THEN BEGIN
  image(TEMPORARY(overflownindex))=overflownvalue
ENDIF

;---------------------------------------------------------------------
; Flip the image if desired (not useful anymore for the PICO images)
;---------------------------------------------------------------------
;orientation=0
;IF (KEYWORD_SET(flip)) THEN BEGIN
;  IF (imageheader.orient NE 0) THEN BEGIN
;  	print,'image does not have to be flipped although specified'
;  ENDIF ELSE BEGIN
;    image=ROTATE(TEMPORARY(image),7)
;    imageheader.orient=(imageheader.orient+1) MOD 2
;    hole(1)=sz(2)-hole(1)
;    imageheader.hole=hole
;    orientation=1
;  ENDELSE
;ENDIF

;*********************************************************************
; If HOLE is known and not 0 then set all the pixels in the hole to 0
; (cosmetics) and shift the image in a way that the hole is centered
; in the flattfielded image.
;*********************************************************************

IF (N_ELEMENTS(hole) EQ 4) THEN BEGIN
  FILL_HOLE,image,HOLE=hole,VALUE=0

  IF ((NOT KEYWORD_SET(nofinalcenter)) AND (N_ELEMENTS(hole) EQ 4)) THEN BEGIN
 	  finalcenter=sz(1:2)/2
    image=IMAGESHIFT(TEMPORARY(image),finalcenter(0:1)-hole(0:1))>0<32000
   	hole(0:1)=finalcenter(0:1)

    IF KEYWORD_SET(fitsheader) THEN BEGIN
      FOR i=0,3 DO SXADDPAR,imageheader,'HOLE'+SC(i),hole(i)
    ENDIF ELSE BEGIN
      imageheader.hole=hole
    ENDELSE

  ENDIF
ENDIF

;*********************************************************************
; Determination of the sky brightness [ppm] in Continuum images;
; Only if the image will be saved and the hole position is known however...
;*********************************************************************
; this has to be changed from the old to the new version. Therefore
; keep the oldversion for the moment and check them first for the
; PICO images as well.

IF KEYWORD_SET(fitsheader) THEN BEGIN
  UNDEFINE,SKYFIT(image,imageheader,COEFF=coeff, OVEROCC=40)
; If SKYFIT does not converge, then coeff is identical 0

  IF (TOTAL(ABS(coeff)) GT 0) THEN BEGIN
    SXADDPAR,imageheader,'SKYFIT0',coeff(0)*SXPAR(imageheader,'CT2PPM')
    SXADDPAR,imageheader,'SKYFIT1',coeff(1)*SXPAR(imageheader,'CT2PPM')
    SXADDPAR,imageheader,'SKYFIT2',coeff(2)
    SXADDPAR,imageheader,'SKYSUN',(coeff(0)+coeff(1))*SXPAR(imageheader,'CT2PPM')
    SXADDPAR,imageheader,'SKY1.2',(coeff(0)+coeff(1)*EXP(50/coeff(2)))*SXPAR(imageheader,'CT2PPM')

    IF KEYWORD_SET(verbose) THEN BEGIN
      MESSAGE,/INFO,/CONT,'Sky brightness '+ $
              SC(SXPAR(imageheader,'SKYSUN'),SIG=3)+' ppm at the limb, '+ $
              SC(SXPAR(imageheader,'SKY1.2'),SIG=3)+' ppm in 1.2Rs; Filter: '+ $
              SXPAR(imageheader,'FILTER')
    ENDIF
  ENDIF

ENDIF ELSE BEGIN
  IF ((N_ELEMENTS(imageheader) NE 0) AND (N_ELEMENTS(flatheader) NE 0) $
	     AND (N_ELEMENTS(hole) EQ 4)) THEN BEGIN
    d=[imageheader.filter,imageheader.polarizer]
    whitecont=(d(0) EQ 5) AND (d(1) EQ 2)
    IRcont=(d(0) EQ 2) AND (d(1) EQ 5)
    Halphacont=(d(0) EQ 5) AND (d(1) EQ 4)

    IF (whitecont OR IRcont OR Halphacont) THEN BEGIN
	    cuthelp=(image(*,hole(1))+image(hole(0),*))/2.
	    cut=(cuthelp(512:1023)+ROTATE(cuthelp(0:511),2))/2.
	    cut=cut*imageheader.ct2ppm
	    imageheader.sky=cut(FIX(FINDGEN(10)*24.3+11*24.3))
    ENDIF
  ENDIF
ENDELSE

;*********************************************************************
; Do the levelling to cut the hot pixels in the flatted image
;*********************************************************************

IF KEYWORD_SET(level) THEN BEGIN
  image=CUT_MAX(TEMPORARY(image),LEVEL=level)
  ADDCOMMENT,imageheader,'FLATTING: Automatical levelling to '+$
    	SC(level*100,DECIMALS=3)+'%'
ENDIF

;*********************************************************************
; Insert the solar grid in the images
;*********************************************************************

IF KEYWORD_SET(grid) THEN BEGIN
  grid0=0
  SUNGRID,HEADER=imageheader, GRID=grid0
  image(WHERE(grid0 GT 0))=MAX(image)
ENDIF

;*********************************************************************
; Do the annotation of the image
;*********************************************************************

IF KEYWORD_SET(annotate) THEN BEGIN
  image=HOLEANNOTATE(TEMPORARY(image),imageheader,/NOKEEP)
ENDIF

;*********************************************************************
; and finally save the resulting image.
; This should only be used with PICO images, since it has not been
; modified for general use epecially for MICA images.
;*********************************************************************

IF (save) THEN BEGIN
  origname=FULL_NAME(STRMID(STRCOMPRESS(imageheader.filename,/REMOVE_ALL),0,4))
  savename=SC(savenr)+'F'+STRMID(origname,STRLEN(origname)-7,3)+'.INT'
  imageheader.filename=STRUPCASE(savename)

  EPHEMERIS,HEADER=imageheader,/MODIFY_HEADER

  PUT_IMG, image, imageheader, DIRECTORY=workdir, $
  	/OVERWRITE, NOKEEP=nokeepimage
ENDIF

ende:
IF KEYWORD_SET(verbose) THEN BEGIN
    MESSAGE,/CONTINUE,/INFORMATIONAL,SC((systime(1)-time),decimals=1)+' sec'
    FLUSH,-1
ENDIF

RETURN,image
END
