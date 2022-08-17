function Automatic, event=event     ;, TrackMode, TrackDiscon, FuncName

COMMON TEMPCTRL, TempFilt, TempStru, FreqFilt, FreqStru
COMMON Paths, PathUsr
COMMON stop_id, sb
COMMON status_text, sb1, FlagHKD, FlagFST

; ----- To Remember !!!! ------
  ; defbat(0) = Default Program
  ; defbat(1) = Time of beginning
  ; defbat(2) = Time of Ending
  ; defbat(3) = 2nd Program
  ; defbat(4) = Time of Beginning
  ; defbat(5) = Time of Ending
  ; defbat(6) = Status Flag
  ; defbat(7) = Action Flag
; -----------------------------

  h1 = systime(1)

; ----------------- Reading Actions.txt file ------------------
  defbat = 0  &  defbat = BatRead()
  NewDefB= 0  &  NewDefB = defbat
  Decision = 0

; -------------------------------------------------------------
; ------------------ Track Analysis ---------------------------
; -------------------------------------------------------------
  a = call_external('track32.dll','GetErrorTrack_')  		; Reset if Error
  IF (a NE 0) THEN BEGIN
     print,"Track Error: ", a    				; SEE WHAT TO DO     		
  ENDIF                                	
  TrkPar = TrkReadP() 	                                 	;  RS 3.3.97
  track = call_function(TrkPar(2), TrkPar(0), TrkPar(1))        ;  RS 3.3.97
  ;if (track eq 99) then defbat(6) = '0'                        ; por ahora ....OJO OJO...


; -------------------------------------------------------------
; ------ Sending of Signal to show that Mica is alive ---------
; -------------------------------------------------------------
  ao = call_external('micro32.dll','Alarm_', long(0), long(0))  

; -------------------------------------------------------------
; -------------------- Weather Station ------------------------

ws = WSTATION()  	; Last Lecture of the Weather Station

; Carlos: PONER ACA QUE SUENE LA ALARMA EN CASO DE VIENTO MUY ALTO
;         O APAGARLA SI SE RESTATURO ( WS(6) = WindSpeed creo)

;;;;;  IF (WS(6) LT 16) THEN ao = call_external('micro32.dll','Alarm_', long(-4), long(0)) 


; -------------------------------------------------------------
; ---------------- HouseKeeping Microcontroller ---------------
; -------------------------------------------------------------
  HKD4 = HousKeep('4')    		; Sky Tester
  wait, .13
  HKD5 = HousKeep('5')			; Sun Tester
  wait, .13

  bb1 = STRING(FORMAT='("SkyT (DU): ",(I3), "   (V) : ",(F6.3))', HKD4(0), HKD4(1))
  WIDGET_CONTROL, sb1(3), set_value = bb1 , /no_copy, sensitive = 1

  bb1 = STRING(FORMAT='("SunT (DU): ",(I3), "   (V) : ",(F6.3))', HKD5(0), HKD5(1))
  WIDGET_CONTROL, sb1(4), set_value = bb1 , /no_copy, sensitive = 1


  IF ( (HKD4(1) NE 0) AND (HKD5(1) NE 0) ) THEN BEGIN  ; Write only when telescope is pointing the sky at least

     FNSky = namecatalog(/sky)			      
     OPENU, 10, FNSky(0), /append
        PRINTF, 10, SC(HKD4(1), sig = 5), '  ',SC(HKD5(1), sig = 5), '  ', strmid(SYSTIME(0),11,8)
     CLOSE, 10

;;;;;  IF (WS(5) GE 25) THEN ao = call_external('micro32.dll','Alarm_', long(4), long(0)) $

  ENDIF

; ................ RS am 06.Mar. 97 ..................
;  HKD8 = HousKeep('8')    	; TEMP 4 segm  1/2
;  HKD9 = HousKeep('9')    	; TEMP 5 segm  3/4
;  HKD10 = HousKeep('10')    	; TEMP 6 segm  5/6
;  HKD11 = HousKeep('11')    	; TEMP 7 segm  7/8
;  HKD12 = HousKeep('12')    	; Filter Ref
;  HKD13 = HousKeep('13')    	; Structure Ref
; .............. END RS am 06.Mar. 97 .................


; -------------------------------------------------------------
; --------------- Filters and Structure Temp. -----------------
; -------------------------------------------------------------
  Temp = float(ReadTemp(10))     ; (Average last 10)
  Temp = (Temp/25.)-273.15

  bb1 = STRING(FORMAT='("Filt/Struct. Temp: ",(F5.2)," °C / ",(F5.2)," °C")', Temp(0), Temp(1))
  WIDGET_CONTROL, sb1(5), set_value = bb1 , /no_copy, sensitive = 1

     IF (Temp(0) LT 34.2) OR (Temp(0) GT 40.0) THEN BEGIN

        TempFilt=0.
        OPENR,10, !MICA.RAMD+"LastFilt.tmp"
          READF,10, TempFilt, FreqF, VoltageF
        CLOSE,10
        tem = SetTemp(Circuit = 2, Voltage = TempFilt)   ; Filters

        IF (Temp(0) LT !MICA.LimTemFil(0)) OR (Temp(0) GT !MICA.LimTemFil(1)) THEN BEGIN
            ao = call_external('micro32.dll','Alarm_', long(16), long(0)) 
        ENDIF ELSE BEGIN
            ao = call_external('micro32.dll','Alarm_', long(-16), long(0)) 
        ENDELSE

     ENDIF ELSE ao = call_external('micro32.dll','Alarm_', long(-16), long(0)) 
 
     IF (Temp(1) LT 24.5) OR (Temp(1) GT 28.0) THEN BEGIN
        TempStru=0.
        OPENR,10, !MICA.RAMD+"LastStru.tmp"
          READF,10, TempStru, FreqS, VoltageS
        CLOSE,10
  ;;;      tem = SetTemp(Circuit = 1, Voltage = TempStru)   ; Structure
     ENDIF

; -------------------------------------------------------------
; --------------- Analysis Time To Sleep  ---------------------
; -------------------------------------------------------------
  HA=0.  &  DEK=0.  &  RA=0.  &  ST=0.
 ;a=CALL_EXTERNAL('track32.dll', 'AskTeleCoordinates_',HA,DEK,RA,ST)   
  a=CALL_EXTERNAL('track32.dll', 'AskSunCoordinates_',HA,DEK,RA,ST)   
;-----------------------------------------------------
; pongo esto pues a veces aparecen valores extranios 
  cv = 0
  WHILE (DEK LT -25) AND (cv LT 5) DO BEGIN
    print, dek
    HA=0.  &  DEK=0.  &  RA=0.  &  ST=0.
    ;a=CALL_EXTERNAL('track32.dll', 'AskTeleCoordinates_',HA,DEK,RA,ST) 
    a=CALL_EXTERNAL('track32.dll', 'AskSunCoordinates_',HA,DEK,RA,ST) 
    cv = cv+1  
  ENDWHILE       ; ojojojojojojojojojo
;-----------------------------------------------------
  IF (DEK LT (-18)) THEN TimeToSleep = 11.50 ELSE TimeToSleep = 15.05  ;3.70           
  IF (HA GT 12) THEN HA=HA-12 ELSE HA=HA+12
 ;;; IF (HA GE TimeToSleep) THEN FinishTime, HA=HA, DEK=DEK, Decision = Decision, $
 ;;;                                         TimeToSleep=TimeToSleep   ; shifted in 12 hs.
  FinishTime, HA=HA, DEK=DEK, Decision = Decision, $
              TimeToSleep=TimeToSleep   ; shifted in 12 hs.


 ; print,"Time in HKD ( inside Automatic() )=", systime(1)-h1
; ================== END Status Weather and Hardware =========================


; ****************************************************************************
; ***************** Analysis of Action Flag ( defbat(7) ) ********************
; ****************************************************************************

Prog = 'Nada' & Prog2 = 'Nada'

  case 1 of

   (defbat(7) eq '1') or (defbat(7) eq '2') : BEGIN		; Execute Default or 2nd Program in the next loop!

            case defbat(6) of
               '0' : begin ; Def.Prog. is able to begin. Nothing is actually running
                        NewDefB(7) = '901'
                     end
               '1' : begin             ; Def.Prog. is actually running.
               			NewDefB(7) = '901'	; Flag to indicate that in the next loop should start program
               			Decision = 1
					 end
               '2' : begin          ; 2nd Prog. is actually running.
               			NewDEfB(7) = '901'	; Flag to indicate that in the next loop should start program
               			Decision = 1
					 end
			   else: begin 	       ; VER LUEGO BIEN
               			NewDefB(7) = '901' ; Flag to indicate that in the next loop should start program
               			Decision = 1
			      	 end
			endcase
			if (defbat(7) eq 1) then begin ; Name of the Program to run
			    Prog = defbat(0)
			    NewDefB(6) = '2'
			endif else begin
			    Prog = defbat(3)
				NewDefB(6) = '7'
			endelse
			Openw, 14, !MICA.RAMD+"Prog.Aux"
			   printf, 14, Prog
			close, 14

  		 END


   (defbat(7) eq '901') : BEGIN		; Execute right now the Program

 	  ; -- Read the name of the Prog. to execute --
 	    Openr, 14, !MICA.RAMD+"Prog.Aux"
	       readf, 14, Prog
	    close, 14
            NewDefB(7) = '0'    ; Si no lo pongo, empieza a andar en circulos
            defbat = NewDefB      
            x = BatSave(defbat)
		  ; -- Updating the .log file --
            messg = " : Going to execute the *** " + Prog + " *** routine"
            actlogfile, message = strmid(systime(0),11,8) + messg
          ; -- Translating of the .PRG if exists --
            Prog = ExistPrg(Prog)
            IF (Prog EQ 'ERROR_in_Source') THEN BEGIN
               stop        ; ERRORS IN SOURCE CODE --> WHAT SHOULD I DO
            ENDIF
          ; -- Executing the .PRO file --
            Ex = ExistPro(Prog)
            IF (Ex EQ 0) THEN BEGIN      ; .PRO doesn't exist
               NOEXISTPRG,  NewDefB = NewDefB, PROG = Prog
            ENDIF ELSE BEGIN
               ; -- Updating Flags --
               defbat = BatRead()
               NewDefB(7) = '0'    ; Si no lo pongo, empieza a andar en circulos
               NewDefB(6) = defbat(6)
               x = BatSave(defbat)
	    ENDELSE
         END


   (defbat(7) eq '3') : BEGIN		; Execute Default first and then 2nd Program in the next loop!

            case defbat(6) of
               '0' : begin ; Def.Prog. is able to begin. Nothing is actually running
                        NewDefB(7) = '801'
                     end
               '1' : begin             ; Def.Prog. is actually running.
               			NewDefB(7) = '801'	; Flag to indicate that in the next loop should start program
               			Decision = 1
					 end
               '2' : begin          ; 2nd Prog. is actually running.
               			NewDEfB(7) = '801'	; Flag to indicate that in the next loop should start program
               			Decision = 1
					 end
			   else: begin 	       ; VER LUEGO BIEN
               			NewDefB(7) = '801' ; Flag to indicate that in the next loop should start program
               			Decision = 1
			      	 end
			endcase
			NewDefB(6) = '2'
			Openw, 14, !MICA.RAMD+"Prog2.Aux"
			   printf, 14, defbat(0)
			   printf, 14, defbat(3)
			close, 14

  		 END

  (defbat(7) eq '801') : BEGIN			; Execute right now the Program

	  ; Read the name of the Prog. to execute
	    Openr, 14, !MICA.RAMD+"Prog2.Aux"
	      readf, 14, Prog
	      readf, 14, Prog2
	    close, 14
            NewDefB(7) = '0'    ; Si no lo pongo, empieza a andar en circulos
            defbat = NewDefB
            x = BatSave(defbat)
		  ; Updating the .log file with the FIRST Program
            messg = " : Going to execute the *** " + Prog + " *** routine"
            actlogfile, message = strmid(systime(0),11,8) + messg
          ; -- Translating of the .PRG if exists --
            Prog = ExistPrg(Prog)
            IF (Prog EQ 'ERROR_in_Source') THEN BEGIN
               stop           ;ERRORS IN SOURCE CODE --> WHAT SHOULD I DO
            ENDIF
          ; -- Executing the .PRO file --
            Ex = ExistPro(Prog)
            IF (Ex EQ 0) THEN BEGIN    ; .PRO doesn't exist
               ; NOEXISTPRG,  NewDefB = NewDefB, PROG = Prog
	       RESULT = 0
	    ENDIF ELSE RESULT = 1
	; 2nd Program to run if not aborted with defbat(7) = 4 or 5
  	    defbat = BatRead()
  	    defbat(6) = '7'
            x = BatSave(defbat)
            if (defbat(7) eq 0) then begin
          ; Updating the .log file with the SECOND Program
               messg = " : Going to execute the *** " + Prog2 + " *** routine"
               actlogfile, message = strmid(systime(0),11,8) + messg
          ; -- Translating of the .PRG if exists --
               Prog2 = ExistPrg(Prog2)
               IF (Prog2 EQ 'ERROR_in_Source') THEN BEGIN
                  stop         ;ERRORS IN SOURCE CODE --> WHAT SHOULD I DO
	       ENDIF
          ; -- Executing the .PRO file --
               Ex = ExistPro(Prog2)
               IF (Ex EQ 0) THEN BEGIN
                  NOEXISTPRG,  NewDefB = NewDefB, PROG = Prog
                  RESULT = 0
	       ENDIF ELSE RESULT = 1
          ; -- Updating Flags --
            endif
            IF (RESULT EQ 1) THEN BEGIN
               defbat = BatRead()
               NewDefB(7) = '0'    ; Si no lo pongo, empieza a andar en circulos
               NewDefB(6) = defbat(6)
               x = BatSave(defbat)
            ENDIF
         END


  (defbat(7) eq '4') : BEGIN		; Stop it right now and set Status to '30'
         Decision = 1			; Aborted by user from MICA interface
         END

  (defbat(7) eq '5') : BEGIN		; Program stopped because ending time
         actlogfile, message = strmid(systime(0),11,8) + " : Program Finished normally (Ending Time)"
         NewDefB(6)='0'
         NewDefB(7)='0'
         Decision = 1
         END

  (defbat(7) eq '15') : BEGIN		; Program stopped from Lindau
           Decision = 1
           defbat = BatRead()          
           defbat(7) = '5'              ; To simulate ending time
           NewDefB(7) = '5'             
           x = BatSave(defbat)         
           actlogfile, message = strmid(systime(0),11,8) + " : Program Stopped from Lindau"
                        END

  (defbat(7) eq '0') and (defbat(6) eq '0') : BEGIN   ; Analize time of beginning (if it's not executing anything)

	TimeComp = strmid(systime(0),11,2) * 60 + strmid(systime(0),14,2)
	if (strlen(defbat(1)) eq 5) then TimeI1 = strmid(defbat(1),0,2) * 60 + strmid(defbat(1),3,2) $
			 	    else TimeI1 = 0
	if (strlen(defbat(2)) eq 5) then TimeF1 = strmid(defbat(2),0,2) * 60 + strmid(defbat(2),3,2) $
				    else TimeF1 = 0
	if (strlen(defbat(4)) eq 5) then TimeI2 = strmid(defbat(4),0,2) * 60 + strmid(defbat(4),3,2) $
				    else TimeI2 = 0
	if (strlen(defbat(5)) eq 5) then TimeF2 = strmid(defbat(5),0,2) * 60 + strmid(defbat(5),3,2) $
					    else TimeF2 = 0

       HA=0.  &  DEK=0.  &  RA=0.  &  ST=0.
       a=CALL_EXTERNAL('track32.dll', 'AskSunCoordinates_',HA,DEK,RA,ST)   
       IF (HA GT 12) THEN HA=HA-12 ELSE HA=HA+12		; shifted in 12 hs.

	; --- Default Prog. ---
	x1 = 0
        IF (HA LT TimeToSleep-.5) THEN BEGIN
          if (TimeI1 gt 0) then begin           ; Analize if it is able to start Default Prog
            if (TimeF1 gt TimeComp) or (TimeF1 eq 0) then begin     ; es mas temprano que la hora de finalizacion
                if (TimeComp ge TimeI1) then x1 = 1	; Flag to indicate that in the next loop should start program
	    endif
	  endif
	ENDIF

	; --- 2nd Prog. ---
	x2 = 0
	IF (HA LT TimeToSleep-.5) THEN BEGIN
          if (TimeI2 gt 0) then begin           ; Analize if it is able to start Default Prog
            if (TimeF2 gt TimeComp) or (TimeF2 eq 0) then begin     ; es mas temprano que la hora de finalizacion
                if (TimeComp ge TimeI2) then x2 = 1	; Son las 10 y deberia haber empezado a las 9 por ej
	    endif
	  endif
	ENDIF

        case 1 of
           	(x1 eq 1) and (x2 eq 0): begin
           				   xx = 1
           				   NewDefB(6) = '1'
           				   Prog = defbat(0)
           				 end
		(x1 eq 0) and (x2 eq 1): begin
				   	   xx = 1
        				   NewDefB(6) = '6
        				   Prog = defbat(3)
        				 end
		(x1 eq 1) and (x2 eq 1): begin
					   xx = 1
			                   Dif1 = TimeComp - TimeI1
                                           Dif2 = TimeComp - TimeI2
                                           If (Dif1 le Dif2) then begin
           				      NewDefB(6) = '1'
           				      Prog = defbat(0)
           				   endif else begin
             				      NewDefB(6) = '6
            				      Prog = defbat(3)
					   endelse
					 end
		else : xx = 0
	endcase

	if (xx gt 0) then begin
	   NewDefB(7) = '901'
	   Openw, 14, !MICA.RAMD+"Prog.Aux"
	      printf, 14, Prog
	   close, 14
	endif

        				      END


  (defbat(7) EQ '0') AND (defbat(6) EQ '1') : BEGIN  ;See Time of Ending Default Program

	TimeComp = strmid(systime(0),11,2) * 60 + strmid(systime(0),14,2)
	if (strlen(defbat(2)) eq 5) then TimeF1 = strmid(defbat(2),0,2) * 60 + strmid(defbat(2),3,2) $
				    else TimeF1 = 0
        HA=0.  &  DEK=0.  &  RA=0.  &  ST=0.
        a=CALL_EXTERNAL('track32.dll', 'AskSunCoordinates_',HA,DEK,RA,ST)   
        IF (HA GT 12) THEN HA=HA-12 ELSE HA=HA+12		; shifted in 12 hs.

	IF ( (TimeF1 LE TimeComp) AND (TimeF1 NE 0) ) OR $
           ( HA GT TimeToSleep-.2 ) THEN BEGIN 
           Decision = 1
           defbat = BatRead()          
           defbat(7) = '5'    ;4         
           NewDefB(7) = '5'   ;4          
         ;;  actlogfile, message = strmid(systime(0),11,8) + " : Program Finished normally (Ending Time)"
           x = BatSave(defbat)         
        ENDIF
					      END


  (defbat(7) EQ '0') AND (defbat(6) EQ '6') : BEGIN  ;See Time of Ending 2nd Program

	TimeComp = strmid(systime(0),11,2) * 60 + strmid(systime(0),14,2)
	if (strlen(defbat(5)) eq 5) then TimeF2 = strmid(defbat(5),0,2) * 60 + strmid(defbat(5),3,2) $
				    else TimeF2 = 0
        HA=0.  &  DEK=0.  &  RA=0.  &  ST=0.
        a=CALL_EXTERNAL('track32.dll', 'AskSunCoordinates_',HA,DEK,RA,ST)   
        IF (HA GT 12) THEN HA=HA-12 ELSE HA=HA+12		; shifted in 12 hs.

	IF ( (TimeF2 LE TimeComp) AND (TimeF2 NE 0) ) OR $
	   ( HA GT TimeToSleep-.2 ) THEN BEGIN
           Decision = 1
           defbat = BatRead()          
           defbat(7) = '5'    ;4   
           NewDefB(7) = '5'   ;4          
         ;;  actlogfile, message = strmid(systime(0),11,8) + " : Program Finished normally (Ending Time)"
           x = BatSave(defbat)         
        ENDIF

      	 				      END


  else : BEGIN  ;ANALIZAR AQUI EN BASE A LA HORA DE INICIO O DE FIN DE ACUERDO AL STATUS FLAG

		 END

endcase


defbat = NewDefB
x = BatSave(defbat)


; ----- Setting of Decision --> It's a number which tells to the camera32 routine
; --------------------- if it is able to continue or not ------------------------
  ;decision = defbat(3)    ; si 1 --> Parar!! / si 0 --> seguir normalmente
                          ; si 2 --> Parar y salir del IDL, cambiando a 1 !MICA.RAMD+cargafg.txt

; ---------- Analysis whether the User pressed the CAMERA STOP Button -----------
 IF KEYWORD_SET(event) THEN BEGIN				
    result = WIDGET_EVENT(event,/nowait,/save)
    IF (result.id EQ event) THEN begin
       decision = 1
       defbat = BatRead()          ; To be able to stop a program by pressing 'Camera Stop Button'
       IF (defbat(7) NE 5) AND (defbat(7) NE 15) THEN BEGIN
          defbat(6) = '30'  
          defbat(7) = '4'   
       ENDIF
       x = BatSave(defbat)  
    ENDIF
 ENDIF

IF (Decision EQ 1) THEN Door, status=1   ;;14/09/97

  return, [decision,track,HKD4(1),HKD5(1)]

end
