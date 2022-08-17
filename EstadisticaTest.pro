PRO EstadisticaTest

;Dir='C:\MICA_Instrument_Tes\'
Dir='C:\Carlos\k-cor Laura Hebe\DatosMeteorológicos\TesOut\'
;DirOut='C:\MICA_Instrument_Tes\'
DirOut='C:\Carlos\k-cor Laura Hebe\DatosMeteorológicos\TesOut\'
anio=1997
fileout=DirOut+'Stat_'+STRTRIM(STRING(anio),2)+'.txt'

file=STRTRIM(STRING(anio),2)+'*.txt'
;file=STRMID(STRTRIM(STRING(anio),2),2,2)+'*.txt'
print, Dir+file
Result=FINDFILE(Dir+file, COUNT=count)
print, count
openw, Unit2, fileout, /GET_LUN
;20000105   11.31   16.64    5.33    9.81   40.81   23.81   25.56
printf, Unit2,'  DATE    T-ini  T-end  %SUN   %CLOU  %GOOD  %MOD   %BAD   OBS'

for i=0, count-1 do begin
    totalmuymalo=0.   ;sun < 2
    totalbueno=0.     ;sky < 1 y sun > 2
    totalregular=0.   ;sun >2 y 1 < sky <2
    totalmalo=0.      ;sun >2 y sky > 2
    totalsun=0.       ;sun >2


    diaout=strmid(Result(i),STRLEN(Result(i))-12,8)
    print, Result(i)
    OPENR, Unit, Result(i), /GET_LUN
    line='                                                                               '
    READF, Unit, line
    contador=0

    WHILE ~ EOF(Unit) DO BEGIN
        READF, Unit, line
        contador+=1
    ENDWHILE

    free_lun, Unit
;---------------------------------------------------------------------------

    ;print, 'Cantidad de lineas de datos ',contador
    cantdat=contador

    horas=FLTARR(contador)
    sky=fltarr(contador)
    sun=fltarr(contador)




    OPENR, Unit, Result(i), /GET_LUN
    line='                                                                               '
    READF, Unit, line


    ;WHILE ~ EOF(Unit) DO BEGIN
    for j=0, cantdat-1 do begin
        READF, Unit, line
        Result2 = STRSPLIT( line , ' ', COUNT=countador,  /EXTRACT)
        if countador eq 3 then begin
            horas(j)=FLOAT(Result2(0))
            sky(j)=FLOAT(Result2(1))
            sun(j)=FLOAT(Result2(2))
        endif
     endfor
    free_lun, Unit

    horainicial=horas(0)
    if horainicial lt 10. then horainicial=10.
    horafinal=horas(cantdat-1)
    if horafinal gt 22. or horafinal lt  horainicial then horafinal=22.
    difhoraria=horafinal-horainicial
    puntos=0. & pmm=0. & pb=0. & pr=0. & pm=0. & ps=0.
    for k=1, cantdat-1 do begin
      if (horas(k) lt 10. or horas(k) gt 22.) then goto, siga
        puntos=puntos+1.
        difhor=ABS(horas(k)-horas(k-1))
        ;if difhor gt 0.001 then begin
        ; sumhor=difhor-0.001
        ;    difhor=0.001
        ;endif else begin
        ;    sumhor=0.
        ;endelse

        if sun(k) ge 2. then begin
            totalsun=totalsun+difhor
            ps=ps+1.
        endif

        if sun(k) lt 2. then begin
            totalmuymalo=totalmuymalo+difhor;+sumhor
            pmm=pmm+1.
            goto, siga
        endif
        if sun(k) ge 2. and sky(k) le 1. then begin
            totalbueno=totalbueno+difhor
            pb=pb+1.
            goto, siga
        endif

        if sun(k) ge 2. and sky(k) gt 1. and sky(k) le 2. then begin
            totalregular=totalregular+difhor
            pr=pr+1.
            goto, siga
        endif

        if sun(k) ge 2. and sky(k) gt 2. then begin
            totalmalo=totalmalo+difhor;+sumhor
            pm=pm+1.
        endif
        siga:
    ;totalmuymalo=0.       ;sun < 2
    ;totalbueno=0.   ;sky < 1 y sun > 2
    ;totalregular=0.     ;sun >2 y 1 < sky <2
    ;totalmalo=0.        ;sun >2 y sky > 2
    ;totalsun=0.
    endfor
    obs='  1'
    tsun=(totalsun/difhoraria)*100.
    tb=(totalbueno/difhoraria)*100.
    tr=(totalregular/difhoraria)*100.
    tm=(totalmalo/difhoraria)*100.
    tmm=(totalmuymalo/difhoraria)*100.

;OJO AQUI HAY PROBLEMA    el 6 12 1997
;     sumacielos=tb+tr+tm
;     if ABS(tsun - sumacielos) gt 0.1 then begin
;        tsun=sumacielos
;        tmm=100.-ABS(tsun)
;     endif


    sum1=(tsun+tmm)/100.
    if sum1 gt 1. then begin
        tsun=tsun/sum1
        tmm=tmm/sum1
    endif
    sum2=(tb+tr+tm)/100.
    if sum2 gt 1. then begin
        tb=tb/sum2
        tr=tr/sum2
        tm=tm/sum2
    endif

    printf,Unit2, format='( A8, 7F7.2, A4)',diaout, horainicial, horafinal, tsun, tmm, tb,  tr, tm , obs

;    printf,Unit2, format='( A8, 6F8.2, A4)',diaout, horainicial, horafinal, (pb/puntos)*100., $
;                                             (pr/puntos)*100., (pm/puntos)*100., (pmm/puntos)*100. , obs
;     print, (pr + pm+ pb+pmm)/puntos
    endfor
;FREE_LUN, Unit
FREE_LUN, Unit2
;plot, horas, sky
;oplot, horas, sun



final:
print, 'TERMINO'
END